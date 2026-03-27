import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm

# Import our updated logic directly from the v5 training script
from train_v5 import (
    OffroadDataset, NUM_CLASSES, CLASS_NAMES, 
    predict_tta, compute_iou, compute_map
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # The root directory itself contains the test 'Color_Images' and 'Segmentation' folders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Passing script_dir to OffroadDataset maps directly to root/Color_Images and root/Segmentation
    test_dataset = OffroadDataset(script_dir, is_train=False)
    test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    print("Loading SegFormer-B4 v5 best checkpoint...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b4-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True)
    
    best_path = os.path.join(script_dir, 'runs', 'best_segformer_b4_v5.pth')
    if not os.path.exists(best_path):
        print(f"Error: Could not find model weights at {best_path}")
        return

    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to(device)
    model.eval()

    iou_accum       = {c:[] for c in range(NUM_CLASSES)}
    all_probs_list  = []
    all_labels_list = []

    print("\nRunning TTA Evaluation on OFFICIAL TEST Dataset...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Test Eval"):
            images, masks = images.to(device), masks.to(device)
            with autocast(device_type='cuda'):
                # Multi-Scale Test Time Augmentation
                outputs = predict_tta(model, images)
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            preds = outputs.argmax(dim=1)
            for c, iou in enumerate(compute_iou(preds, masks, NUM_CLASSES)):
                if not np.isnan(iou):
                    iou_accum[c].append(iou)
            
            probs = F.softmax(outputs, dim=1)
            B, NC, H, W = probs.shape
            pf = probs.permute(0,2,3,1).reshape(-1,NC)
            lf = masks.reshape(-1)
            vm = lf != 255
            all_probs_list.append(pf[vm].cpu().numpy())
            all_labels_list.append(lf[vm].cpu().numpy())

    final_ious = [np.mean(iou_accum[c]) if iou_accum[c] else 0.0 for c in range(NUM_CLASSES)]
    final_mIoU = float(np.nanmean(final_ious))

    all_probs_np  = np.concatenate(all_probs_list, axis=0)
    all_labels_np = np.concatenate(all_labels_list, axis=0)
    
    # Cap size for mAP calculation to avoid memory limits
    if len(all_labels_np) > 500000:
        idx = np.random.choice(len(all_labels_np), 500000, replace=False)
        all_probs_np  = all_probs_np[idx]
        all_labels_np = all_labels_np[idx]
        
    final_mAP = compute_map(all_probs_np, all_labels_np, NUM_CLASSES)

    print(f"\n{'='*55}")
    print(f"OFFICIAL TEST SET RESULTS (TTA)")
    print(f"mIoU : {final_mIoU:.4f}")
    print(f"mAP  : {final_mAP:.4f}")
    print("="*55)
    for name, iou in zip(CLASS_NAMES, final_ious):
        print(f"  {name:<15}: {iou:.4f}")

if __name__ == '__main__':
    main()