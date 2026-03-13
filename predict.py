import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm

# Import from our custom dataset module
from dataset import OffroadTestDataset, NUM_CLASSES
from utils import CLASS_NAMES

# The exact color palette from the baseline visualization
COLOR_PALETTE = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

def mask_to_color(mask):
    """Convert a 2D class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages', 'Offroad_Segmentation_testImages', 'Color_Images')
    model_path = os.path.join(script_dir, 'runs', 'best_segformer_b2.pth')
    
    output_raw_dir = os.path.join(script_dir, 'predictions', 'raw_masks')
    output_color_dir = os.path.join(script_dir, 'predictions', 'color_masks')
    os.makedirs(output_raw_dir, exist_ok=True)
    os.makedirs(output_color_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading unseen test images from: {test_dir}")
    test_dataset = OffroadTestDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 2. Load Model
    print(f"Loading best model weights from: {model_path}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Inference Loop
    print("\nStarting inference on unseen environment...")
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            
            with autocast(device_type='cuda'):
                outputs = model(pixel_values=images).logits
                
                # Upsample back to 512x512
                outputs = F.interpolate(
                    outputs,
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False
                )
            
            # Get predicted class IDs (0-9)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
            
            # Save raw and colorized masks
            for i, filename in enumerate(filenames):
                pred_mask = predictions[i]
                
                # Save raw mask (for submission/evaluation)
                raw_path = os.path.join(output_raw_dir, filename)
                cv2.imwrite(raw_path, pred_mask)
                
                # Save colorized mask (for report visualization)
                color_mask = mask_to_color(pred_mask)
                color_path = os.path.join(output_color_dir, filename)
                # Convert RGB to BGR for cv2 saving
                cv2.imwrite(color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    print("\nInference Complete!")
    print(f"Raw masks saved to:   {output_raw_dir}")
    print(f"Color masks saved to: {output_color_dir}")

if __name__ == "__main__":
    main()