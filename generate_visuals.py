import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

# Import the correct 7-class dataset and mapping directly from train_v5
from train_v5 import OffroadDataset, NUM_CLASSES, CLASS_NAMES

COLOR_PALETTE = np.array([
    [34, 139, 34],    # 0: Trees
    [0, 255, 0],      # 1: Lush Bushes
    [210, 180, 140],  # 2: Dry Grass
    [139, 90, 43],    # 3: Dry Bushes
    [128, 128, 128],  # 4: Rocks (Merged from Ground Clutter)
    [160, 82, 45],    # 5: Landscape (Merged from Logs)
    [135, 206, 235],  # 6: Sky
], dtype=np.uint8)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_dir = os.path.join(script_dir, 'val')
    out_dir = os.path.join(script_dir, 'report_visuals')
    os.makedirs(out_dir, exist_ok=True)

    dataset = OffroadDataset(val_dir, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model with exactly 7 classes
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b4-finetuned-ade-512-512",
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    
    model_path = os.path.join(script_dir, 'runs', 'best_segformer_b4_v5.pth')
    
    # weights_only=True added to fix the future warning
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    print("Generating Failure Case & Success Visuals for Report...")
    saved_count = 0

    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            if saved_count >= 10: break

            image = image.to(device)
            mask = mask.to(device).squeeze(0).cpu().numpy()
            
            outputs = model(pixel_values=image).logits
            outputs = F.interpolate(outputs, size=mask.shape, mode='bilinear', align_corners=False)
            pred = outputs.argmax(dim=1).squeeze(0).cpu().numpy()

            # Calculate error map (where prediction != ground truth)
            valid = mask != 255
            error_map = np.zeros_like(mask, dtype=np.uint8)
            error_map[valid & (pred != mask)] = 1  # 1 indicates error

            # Save interesting cases (where there are some errors to discuss)
            if error_map.sum() > 5000:
                img_vis = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_vis = ((img_vis * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
                
                gt_color = mask_to_color(mask)
                pred_color = mask_to_color(pred)

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img_vis)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                axes[1].imshow(gt_color)
                axes[1].set_title("Ground Truth (7 Classes)")
                axes[1].axis('off')
                
                axes[2].imshow(pred_color)
                axes[2].set_title("v5 Prediction")
                axes[2].axis('off')

                axes[3].imshow(error_map, cmap='Reds')
                axes[3].set_title("Error Map (Red = Misclassified)")
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'analysis_{saved_count}.png'), dpi=150)
                plt.close()
                saved_count += 1

    print(f"Saved 10 analysis visuals to {out_dir}")

if __name__ == '__main__':
    main()