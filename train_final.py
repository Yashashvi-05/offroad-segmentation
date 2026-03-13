"""
Final push script — SWA + OneCycleLR + TTA evaluation
Target: push mIoU from 0.5392 to 0.65+
Runtime: ~90 minutes on RTX 4050
"""
import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def predict_with_tta(model, images, num_classes=10):
    """
    Test Time Augmentation — average 4 predictions:
    original, hflip, brightness+, brightness-
    Gives free +3-5% mIoU with no retraining.
    """
    with torch.no_grad():
        # Original
        out1 = model(pixel_values=images).logits
        # Horizontal flip
        out2 = model(pixel_values=torch.flip(images, dims=[-1])).logits
        out2 = torch.flip(out2, dims=[-1])
        # Slight brightness increase
        out3 = model(pixel_values=torch.clamp(images * 1.1, -3, 3)).logits
        # Slight brightness decrease
        out4 = model(pixel_values=torch.clamp(images * 0.9, -3, 3)).logits

        # Average all predictions
        avg = (out1 + out2 + out3 + out4) / 4.0
    return avg


def validate_with_tta(model, loader, criterion, device, num_classes=10):
    model.eval()
    val_loss       = 0.0
    val_ious_accum = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val+TTA", leave=False):
            images, masks = images.to(device), masks.to(device)

            with autocast(device_type='cuda'):
                outputs = predict_with_tta(model, images, num_classes)
                outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                        mode="bilinear", align_corners=False)
                loss = criterion(outputs, masks)

            val_loss += loss.item()
            for c, iou in enumerate(compute_batch_iou(outputs, masks, num_classes)):
                if not np.isnan(iou):
                    val_ious_accum[c].append(iou)

    epoch_ious = [np.mean(val_ious_accum[c]) if val_ious_accum[c] else float('nan')
                  for c in range(num_classes)]
    return val_loss / len(loader), float(np.nanmean(epoch_ious)), epoch_ious


def save_charts(ious, mIoU, history, runs_dir):
    # Per-class IoU bar chart
    plt.figure(figsize=(12, 6))
    colors = ['green' if iou >= 0.4 else 'orange' if iou >= 0.2 else 'red'
              for iou in ious]
    plt.bar(CLASS_NAMES, ious, color=colors, edgecolor='black')
    plt.axhline(y=mIoU, color='blue', linestyle='--', linewidth=2,
                label=f'mIoU = {mIoU:.4f}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('IoU Score')
    plt.title('Per-Class IoU — SegFormer-B2 Final (with TTA)')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, 'final_per_class_iou.png'), dpi=150)
    plt.close()

    # Training curve
    if history:
        plt.figure(figsize=(10, 5))
        plt.plot([h['epoch'] for h in history],
                 [h['train_loss'] for h in history], label='Train Loss')
        plt.plot([h['epoch'] for h in history],
                 [h['val_loss'] for h in history], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves — Final Run')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(runs_dir, 'final_loss_curve.png'), dpi=150)
        plt.close()

    print(f"Charts saved to {runs_dir}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- CONFIG ---
    # 15 epochs x 6 min = 90 min — leaves 30 min for TTA eval + charts
    batch_size   = 4
    num_epochs   = 15
    num_classes  = 10
    num_workers  = 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir  = os.path.join(script_dir, 'train')
    val_dir    = os.path.join(script_dir, 'val')
    runs_dir   = os.path.join(script_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    # --- CLASS WEIGHTS ---
    train_masks_dir = os.path.join(train_dir, 'Segmentation')
    print("Calculating class weights...")
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

    # --- MODEL: Load best checkpoint ---
    print("Loading SegFormer-B2 from best checkpoint...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=num_classes, ignore_mismatched_sizes=True)
    checkpoint = os.path.join(runs_dir, "best_segformer_b2.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    print(f"Resumed from: {checkpoint} (baseline mIoU: 0.5392)")

    # --- SWA SETUP ---
    # Stochastic Weight Averaging: maintains average of weights over epochs
    # This smooths the loss landscape and improves generalization
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 5  # start averaging after epoch 5

    # --- OPTIMIZER: OneCycleLR ---
    # Starts at low LR, peaks in middle, decays — proven to escape plateaus
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=6e-5,           # peak LR
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,         # 30% warmup
        div_factor=10,         # start at max_lr/10
        final_div_factor=100   # end at max_lr/100
    )

    criterion = SegmentationLoss(class_weights=class_weights, num_classes=num_classes)
    scaler    = GradScaler(device='cuda')
    logger    = EpochLogger(CLASS_NAMES)

    best_mIoU  = 0.5392
    history    = []

    # --- TRAINING LOOP ---
    print(f"\nStarting final push — {num_epochs} epochs + SWA + TTA\n")

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
            scheduler.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Update SWA model after swa_start
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            print(f"  SWA updated at epoch {epoch}")

        # VALIDATE (regular model)
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
        history.append({'epoch': epoch, 'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss, 'mIoU': mIoU})

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            torch.save(model.state_dict(),
                       os.path.join(runs_dir, "best_segformer_b2.pth"))
            print(f"--> New Best! mIoU: {best_mIoU:.4f} saved.")

    # --- SWA FINAL EVALUATION ---
    print("\nUpdating SWA batch norms...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    print("Evaluating SWA model with TTA...")
    swa_model.to(device)
    val_loss_tta, mIoU_tta, ious_tta = validate_with_tta(
        swa_model, val_loader, criterion, device, num_classes)

    print(f"\n{'='*50}")
    print(f"SWA + TTA Final mIoU: {mIoU_tta:.4f}")
    print(f"{'='*50}")
    for name, iou in zip(CLASS_NAMES, ious_tta):
        print(f"  {name:<20}: {iou:.4f}")

    # Save SWA model if better
    if mIoU_tta > best_mIoU:
        best_mIoU = mIoU_tta
        torch.save(swa_model.state_dict(),
                   os.path.join(runs_dir, "best_segformer_b2.pth"))
        print(f"\nSWA model is best! Saved. mIoU: {best_mIoU:.4f}")
    else:
        print(f"\nKeeping regular best model. mIoU: {best_mIoU:.4f}")

    # Save all charts
    save_charts(ious_tta, mIoU_tta, history, runs_dir)
    print(f"\nFinal Best mIoU: {best_mIoU:.4f}")


if __name__ == "__main__":
    main()