"""
Test evaluator for v5 model — 7 classes, same merging as v4/v5
"""
import torch, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

VALUE_MAP = {
    100:  0, 200:  1, 300:  2, 500:  3,
    550:  4, 600:  2, 700:  5, 800:  4,
    7100: 5, 10000:6,
}

CLASS_NAMES = [
    'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Rocks', 'Landscape', 'Sky'
]
NUM_CLASSES = 7

def remap_mask(mask):
    output = np.full(mask.shape, 255, dtype=np.uint8)
    for raw_val, class_idx in VALUE_MAP.items():
        output[mask == raw_val] = class_idx
    return output

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith('.png')])
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485,0.456,0.406],
                       std=[0.229,0.224,0.225]),
            ToTensorV2()])
        print(f"Test dataset: {len(self.data_ids)} images")

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        fname = self.data_ids[idx]
        image = cv2.imread(os.path.join(self.image_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(
            os.path.join(self.mask_dir, fname),
            cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = remap_mask(mask)
        t = self.transform(image=image, mask=mask)
        return t['image'], t['mask'].long()

def compute_map(all_probs, all_labels, num_classes=7):
    aps = []
    for c in range(num_classes):
        binary_labels = (all_labels==c).astype(int)
        if binary_labels.sum() == 0:
            continue
        try:
            ap = average_precision_score(
                binary_labels, all_probs[:,c])
            aps.append(ap)
        except Exception:
            pass
    return np.mean(aps) if aps else 0.0

def predict_tta(model, images):
    out1 = model(pixel_values=images).logits
    out2 = model(pixel_values=torch.flip(images,[-1])).logits
    out2 = torch.flip(out2, [-1])
    out3 = model(pixel_values=torch.clamp(images*1.15,-3,3)).logits
    out4 = model(pixel_values=torch.clamp(images*0.85,-3,3)).logits
    out5 = model(pixel_values=torch.clamp(images*1.05,-3,3)).logits
    return (out1+out2+out3+out4+out5)/5.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load v5 if exists, else fall back to v4
v5_path = os.path.join(script_dir, 'runs', 'best_segformer_b4_v5.pth')
v4_path = os.path.join(script_dir, 'runs', 'best_segformer_b4_v4.pth')
model_path = v5_path if os.path.exists(v5_path) else v4_path
print(f"Loading model: {os.path.basename(model_path)}")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

test_dir = os.path.join(script_dir, '..',
    'Offroad_Segmentation_testImages',
    'Offroad_Segmentation_testImages')
dataset = TestDataset(test_dir)
loader  = DataLoader(dataset, batch_size=4,
                     shuffle=False, num_workers=0)

all_ious        = {c:[] for c in range(NUM_CLASSES)}
confusion       = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.int64)
all_probs_list  = []
all_labels_list = []

print("Evaluating TEST SET with 5x TTA...")
with torch.no_grad():
    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device)
        with torch.cuda.amp.autocast():
            outputs = predict_tta(model, images)
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:],
                mode='bilinear', align_corners=False)
        preds  = outputs.argmax(dim=1).view(-1).cpu().numpy()
        labels = masks.view(-1).cpu().numpy()
        valid  = labels < NUM_CLASSES
        for c in range(NUM_CLASSES):
            inter = np.sum((preds==c) & (labels==c))
            union = np.sum((preds==c) | (labels==c))
            if union > 0:
                all_ious[c].append(inter/union)
        np.add.at(confusion,(labels[valid],preds[valid]),1)
        probs = F.softmax(outputs, dim=1)
        B,NC,H,W = probs.shape
        pf = probs.permute(0,2,3,1).reshape(-1,NC)
        lf = masks.reshape(-1)
        vm = lf != 255
        all_probs_list.append(pf[vm].cpu().numpy())
        all_labels_list.append(lf[vm].cpu().numpy())

ious = [np.mean(all_ious[c]) if all_ious[c] else 0.0
        for c in range(NUM_CLASSES)]
mIoU = np.mean(ious)

all_probs_np  = np.concatenate(all_probs_list, axis=0)
all_labels_np = np.concatenate(all_labels_list, axis=0)
if len(all_labels_np) > 500000:
    idx = np.random.choice(len(all_labels_np),500000,replace=False)
    all_probs_np  = all_probs_np[idx]
    all_labels_np = all_labels_np[idx]
mAP = compute_map(all_probs_np, all_labels_np, NUM_CLASSES)

# Dominant class mIoU (classes >1% of test pixels)
dominant = {'Dry Grass':ious[2],'Rocks':ious[4],
            'Landscape':ious[5],'Sky':ious[6],'Dry Bushes':ious[3]}
dominant_mIoU = np.mean(list(dominant.values()))

print(f"\n{'='*55}")
print(f"TEST SET RESULTS — 7 Classes")
print(f"mIoU (all 7):     {mIoU:.4f}")
print(f"mIoU (dominant):  {dominant_mIoU:.4f}")
print(f"mAP:              {mAP:.4f}")
print(f"{'='*55}")
for name, iou in zip(CLASS_NAMES, ious):
    flag = " !! LOW" if iou < 0.4 else ""
    print(f"  {name:<15}: {iou:.4f}{flag}")

print(f"\nTest distribution context:")
print(f"  Landscape 43% | Rocks 18% | Sky 18%")
print(f"  DryGrass 17%  | DryBushes 3% | Trees 0.3%")
print(f"  LushBushes 0% — absent from test desert")

# Per-class IoU chart
plt.figure(figsize=(12,6))
colors = ['green' if iou>=0.5 else
          'orange' if iou>=0.3 else 'red'
          for iou in ious]
plt.bar(CLASS_NAMES, ious, color=colors, edgecolor='black')
plt.axhline(y=mIoU, color='blue', linestyle='--',
            linewidth=2,
            label=f'mIoU={mIoU:.4f} | mAP={mAP:.4f}')
plt.axhline(y=dominant_mIoU, color='green',
            linestyle=':', linewidth=2,
            label=f'Dominant mIoU={dominant_mIoU:.4f}')
plt.xticks(rotation=45, ha='right')
plt.ylabel('IoU Score')
plt.title(f'TEST SET — SegFormer-B4 v5\n'
          f'mIoU={mIoU:.4f} | mAP={mAP:.4f}')
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('test_v5_per_class_iou.png', dpi=150)
plt.close()

# Confusion matrix
plt.figure(figsize=(10,8))
conf_norm = confusion.astype(float)
row_sums  = conf_norm.sum(axis=1,keepdims=True)
row_sums[row_sums==0] = 1
conf_norm = conf_norm/row_sums
plt.imshow(conf_norm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(range(NUM_CLASSES),CLASS_NAMES,rotation=45,ha='right')
plt.yticks(range(NUM_CLASSES),CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title(f'TEST SET Confusion Matrix\n'
          f'mIoU={mIoU:.4f} | mAP={mAP:.4f}')
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j,i,f'{conf_norm[i,j]:.2f}',
                 ha='center',va='center',
                 color='white' if conf_norm[i,j]>0.5
                 else 'black',fontsize=9)
plt.tight_layout()
plt.savefig('test_v5_confusion_matrix.png', dpi=150)
plt.close()

print(f"\nSaved: test_v5_per_class_iou.png")
print(f"Saved: test_v5_confusion_matrix.png")
print(f"\nFINAL: mIoU={mIoU:.4f} | mAP={mAP:.4f}")