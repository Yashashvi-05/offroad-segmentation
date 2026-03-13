"""
TTA-only evaluation — no training, no risk.
Runs existing best model with Test Time Augmentation.
Cannot decrease score. Takes ~10 minutes.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm
from dataset import OffroadSegmentationDataset
from utils import CLASS_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load best model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=10, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(
    os.path.join(script_dir, 'runs', 'best_segformer_b2.pth'),
    map_location=device))
model.to(device).eval()

val_dir = os.path.join(script_dir, 'val')
dataset = OffroadSegmentationDataset(val_dir, is_train=False)
loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

all_ious = {c: [] for c in range(10)}

print("Running TTA evaluation...")
with torch.no_grad():
    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device)

        # 4 augmented predictions averaged
        out1 = model(pixel_values=images).logits
        out2 = model(pixel_values=torch.flip(images, dims=[-1])).logits
        out2 = torch.flip(out2, dims=[-1])
        out3 = model(pixel_values=torch.clamp(images * 1.1, -3, 3)).logits
        out4 = model(pixel_values=torch.clamp(images * 0.9, -3, 3)).logits
        outputs = (out1 + out2 + out3 + out4) / 4.0

        outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                mode='bilinear', align_corners=False)
        preds  = outputs.argmax(dim=1).view(-1).cpu().numpy()
        labels = masks.view(-1).cpu().numpy()

        for c in range(10):
            inter = np.sum((preds == c) & (labels == c))
            union = np.sum((preds == c) | (labels == c))
            if union > 0:
                all_ious[c].append(inter / union)

ious = [np.mean(all_ious[c]) if all_ious[c] else 0.0 for c in range(10)]
mIoU = np.mean(ious)

print(f"\nWithout TTA: 0.5392")
print(f"With TTA:    {mIoU:.4f}")
print(f"Improvement: +{(mIoU - 0.5392):.4f}")
print(f"\nPer-class IoU:")
for name, iou in zip(CLASS_NAMES, ious):
    print(f"  {name:<20}: {iou:.4f}")

# Save updated charts
plt.figure(figsize=(12, 6))
colors = ['green' if iou >= 0.4 else 'orange' if iou >= 0.2 else 'red'
          for iou in ious]
plt.bar(CLASS_NAMES, ious, color=colors, edgecolor='black')
plt.axhline(y=mIoU, color='blue', linestyle='--', linewidth=2,
            label=f'mIoU = {mIoU:.4f}')
plt.xticks(rotation=45, ha='right')
plt.ylabel('IoU Score')
plt.title(f'Per-Class IoU — SegFormer-B2 + TTA (mIoU={mIoU:.4f})')
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('per_class_iou_tta.png', dpi=150)
plt.close()
print("\nSaved: per_class_iou_tta.png")