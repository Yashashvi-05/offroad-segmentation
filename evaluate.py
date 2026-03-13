import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from dataset import OffroadSegmentationDataset
from utils import CLASS_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=10, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(
    os.path.join(script_dir, 'runs', 'best_segformer_b2.pth'),
    map_location=device))
model.to(device).eval()

# Load val dataset
val_dir = os.path.join(script_dir, 'val')
dataset = OffroadSegmentationDataset(val_dir, is_train=False)
loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# Evaluate
all_ious     = {c: [] for c in range(10)}
confusion    = np.zeros((10, 10), dtype=np.int64)

with torch.no_grad():
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(pixel_values=images).logits
        outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                mode='bilinear', align_corners=False)
        preds  = outputs.argmax(dim=1).view(-1).cpu().numpy()
        labels = masks.view(-1).cpu().numpy()

        # IoU
        for c in range(10):
            inter = np.sum((preds == c) & (labels == c))
            union = np.sum((preds == c) | (labels == c))
            if union > 0:
                all_ious[c].append(inter / union)

        # Confusion matrix
        valid = labels < 10
        np.add.at(confusion, (labels[valid], preds[valid]), 1)

# Compute final IoUs
ious = [np.mean(all_ious[c]) if all_ious[c] else 0.0 for c in range(10)]
mIoU = np.mean(ious)

print(f"\nmIoU on Validation Set: {mIoU:.4f}\n")
print(f"{'Class':<20} {'IoU':>8}")
print("-" * 30)
for name, iou in zip(CLASS_NAMES, ious):
    print(f"{name:<20} {iou:.4f}")

# --- PLOT 1: Per-class IoU bar chart ---
plt.figure(figsize=(12, 6))
colors = ['green' if iou >= 0.4 else 'orange' if iou >= 0.2 else 'red' for iou in ious]
bars = plt.bar(CLASS_NAMES, ious, color=colors, edgecolor='black')
plt.axhline(y=mIoU, color='blue', linestyle='--', linewidth=2, label=f'mIoU = {mIoU:.4f}')
plt.xticks(rotation=45, ha='right')
plt.ylabel('IoU Score')
plt.title('Per-Class IoU — SegFormer-B2 Final Model')
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('per_class_iou.png', dpi=150)
print("\nSaved: per_class_iou.png")

# --- PLOT 2: Confusion matrix ---
plt.figure(figsize=(12, 10))
conf_normalized = confusion.astype(float)
row_sums = conf_normalized.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
conf_normalized = conf_normalized / row_sums

plt.imshow(conf_normalized, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(range(10), CLASS_NAMES, rotation=45, ha='right')
plt.yticks(range(10), CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Normalized) — SegFormer-B2')
for i in range(10):
    for j in range(10):
        plt.text(j, i, f'{conf_normalized[i,j]:.2f}',
                 ha='center', va='center',
                 color='white' if conf_normalized[i,j] > 0.5 else 'black',
                 fontsize=7)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Saved: confusion_matrix.png")

print("\nAll charts saved. Add these to your report.")