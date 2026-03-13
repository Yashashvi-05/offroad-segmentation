import torch, numpy as np, os, matplotlib
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

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=10, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(
    os.path.join(script_dir, 'runs', 'best_segformer_b2.pth'), map_location=device))
model.to(device).eval()

# TEST SET — actual unseen environment
test_dir = os.path.join(script_dir, '..', 
    'Offroad_Segmentation_testImages', 
    'Offroad_Segmentation_testImages')

dataset = OffroadSegmentationDataset(test_dir, is_train=False)
loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

all_ious  = {c: [] for c in range(10)}
confusion = np.zeros((10, 10), dtype=np.int64)

print("Evaluating on ACTUAL TEST SET...")
with torch.no_grad():
    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(pixel_values=images).logits
        outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                mode='bilinear', align_corners=False)
        preds  = outputs.argmax(dim=1).view(-1).cpu().numpy()
        labels = masks.view(-1).cpu().numpy()
        for c in range(10):
            inter = np.sum((preds==c) & (labels==c))
            union = np.sum((preds==c) | (labels==c))
            if union > 0:
                all_ious[c].append(inter/union)
        valid = labels < 10
        np.add.at(confusion, (labels[valid], preds[valid]), 1)

ious = [np.mean(all_ious[c]) if all_ious[c] else 0.0 for c in range(10)]
mIoU = np.mean(ious)

print(f"\n{'='*50}")
print(f"TEST SET mIoU: {mIoU:.4f}")
print(f"{'='*50}")
for name, iou in zip(CLASS_NAMES, ious):
    flag = " !! LOW" if iou < 0.4 else ""
    print(f"  {name:<20}: {iou:.4f}{flag}")

# Save test charts
plt.figure(figsize=(12,6))
colors = ['green' if iou>=0.4 else 'orange' if iou>=0.2 else 'red' for iou in ious]
plt.bar(CLASS_NAMES, ious, color=colors, edgecolor='black')
plt.axhline(y=mIoU, color='blue', linestyle='--', linewidth=2,
            label=f'mIoU = {mIoU:.4f}')
plt.xticks(rotation=45, ha='right')
plt.ylabel('IoU Score')
plt.title(f'Per-Class IoU on TEST SET — SegFormer-B2 (mIoU={mIoU:.4f})')
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('test_per_class_iou.png', dpi=150)
plt.close()

# Confusion matrix
plt.figure(figsize=(12,10))
conf_norm = confusion.astype(float)
row_sums  = conf_norm.sum(axis=1, keepdims=True)
row_sums[row_sums==0] = 1
conf_norm = conf_norm / row_sums
plt.imshow(conf_norm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(range(10), CLASS_NAMES, rotation=45, ha='right')
plt.yticks(range(10), CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix — TEST SET (mIoU={mIoU:.4f})')
for i in range(10):
    for j in range(10):
        plt.text(j, i, f'{conf_norm[i,j]:.2f}',
                 ha='center', va='center',
                 color='white' if conf_norm[i,j]>0.5 else 'black', fontsize=7)
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=150)
plt.close()

print("\nSaved: test_per_class_iou.png")
print("Saved: test_confusion_matrix.png")
print("\nThese are your REAL submission scores.")