import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm

from dataset import OffroadSegmentationDataset
from losses import SegmentationLoss
from utils import analyze_class_frequencies, EpochLogger, CLASS_NAMES


def compute_batch_iou(preds, labels, num_classes=10, ignore_index=255):
    preds  = preds.argmax(dim=1).view(-1)
    labels = labels.view(-1)
    valid  = labels != ignore_index
    preds  = preds[valid]
    labels = labels[valid]
    ious   = []
    for cls in range(num_classes):
        pred_inds    = preds == cls
        target_inds  = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union        = (pred_inds | target_inds).sum().item()
        ious.append(float('nan') if union == 0 else intersection / union)
    return ious


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- CONFIG ---
    batch_size  = 4
    num_epochs  = 25      # 25 x 6min = 150min = 2.5 hours
    num_classes = 10
    learning_rate = 3e-5  # lower LR for fine-tuning continuation
    num_workers = 0       # Windows requires 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir  = os.path.join(script_dir, 'train')
    val_dir    = os.path.join(script_dir, 'val')
    runs_dir   = os.path.join(script_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    # --- CLASS WEIGHTS ---
    train_masks_dir = os.path.join(train_dir, 'Segmentation')
    print("Calculating class weights from pixel frequency...")
    class_weights = analyze_class_frequencies(train_masks_dir, num_classes=num_classes)
    class_weights = class_weights.to(device)

    # --- DATASETS ---
    print("Setting up datasets...")
    train_dataset = OffroadSegmentationDataset(train_dir, is_train=True)
    val_dataset   = OffroadSegmentationDataset(val_dir,   is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- MODEL: SegFormer-B2 ---
    print("Loading SegFormer-B2...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    # --- RESUME FROM CHECKPOINT ---
    checkpoint = os.path.join(runs_dir, "best_segformer_b2.pth")
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Resumed from checkpoint: {checkpoint}")
        print(f"Continuing from best mIoU: 0.4987")
    else:
        print("No checkpoint found — starting from scratch")

    # --- OPTIMIZER ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # --- LOSS + SCHEDULER + SCALER ---
    criterion = SegmentationLoss(class_weights=class_weights, num_classes=num_classes)

    # CosineAnnealingWarmRestarts — resets LR at epoch 10 to escape plateaus
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    scaler     = GradScaler(device='cuda')
    logger     = EpochLogger(CLASS_NAMES)
    best_mIoU  = 0.4987  # start from known best so we only save improvements
    patience   = 7
    no_improve = 0

    # --- TRAINING LOOP ---
    print(f"\nStarting: SegFormer-B2 resumed | {num_epochs} epochs | batch {batch_size}\n")

    for epoch in range(1, num_epochs + 1):

        # TRAIN
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader,
                          desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)

        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(pixel_values=images).logits
                outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                        mode="bilinear", align_corners=False)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # VALIDATE
        model.eval()
        val_loss       = 0.0
        val_ious_accum = {c: [] for c in range(num_classes)}
        val_pbar = tqdm(val_loader,
                        desc=f"Epoch {epoch}/{num_epochs} [Val]  ", leave=False)

        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)

                with autocast(device_type='cuda'):
                    outputs = model(pixel_values=images).logits
                    outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                            mode="bilinear", align_corners=False)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()
                for c, iou in enumerate(compute_batch_iou(outputs, masks, num_classes)):
                    if not np.isnan(iou):
                        val_ious_accum[c].append(iou)

        avg_val_loss = val_loss / len(val_loader)
        epoch_ious   = [np.mean(val_ious_accum[c]) if val_ious_accum[c] else float('nan')
                        for c in range(num_classes)]
        mIoU = float(np.nanmean(epoch_ious))

        logger.log_epoch(epoch, avg_train_loss, avg_val_loss, epoch_ious, mIoU)

        # SAVE BEST
        if mIoU > best_mIoU:
            best_mIoU  = mIoU
            no_improve = 0
            save_path  = os.path.join(runs_dir, "best_segformer_b2.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best! mIoU: {best_mIoU:.4f} saved.")
        else:
            no_improve += 1
            print(f"    No improvement {no_improve}/{patience}. Best: {best_mIoU:.4f}")

        # EARLY STOPPING
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best mIoU: {best_mIoU:.4f}")
            break

    print(f"\nDone. Best mIoU: {best_mIoU:.4f}")
    print(f"Model: {os.path.join(runs_dir, 'best_segformer_b2.pth')}")


if __name__ == "__main__":
    main()