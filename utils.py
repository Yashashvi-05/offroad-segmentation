import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Complete mapping - NOTE: 600 (Flowers) was missing in the original, fixed here
VALUE_MAP = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers  <-- THIS WAS MISSING
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9   # Sky
}

CLASS_NAMES = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

NUM_CLASSES = 10


def remap_mask(mask):
    """
    Safely remaps raw mask pixel values to class indices 0-9.
    Uses a dictionary lookup approach to safely handle large pixel values
    like 7100 and 10000 without creating massive arrays.
    """
    output = np.full(mask.shape, 255, dtype=np.uint8)  # 255 = ignore index
    for raw_val, class_idx in VALUE_MAP.items():
        output[mask == raw_val] = class_idx
    return output


def analyze_class_frequencies(mask_dir, num_classes=NUM_CLASSES):
    """
    Scans all masks in the training directory, counts pixel frequencies,
    and returns normalized weights for CrossEntropyLoss.
    
    Args:
        mask_dir: Path to the folder containing segmentation mask .png files
        num_classes: Number of segmentation classes (default 10)
    
    Returns:
        torch.FloatTensor of shape (num_classes,) with loss weights
    """
    print(f"\nScanning masks in: {mask_dir}")

    mask_files = [f for f in os.listdir(mask_dir)
                  if f.lower().endswith('.png')]

    if not mask_files:
        raise FileNotFoundError(f"No PNG mask files found in {mask_dir}")

    pixel_counts = np.zeros(num_classes, dtype=np.float64)

    for filename in tqdm(mask_files, desc="Analyzing Masks"):
        filepath = os.path.join(mask_dir, filename)
        # Read as unchanged to preserve raw pixel values
        mask = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"  WARNING: Could not read {filename}, skipping.")
            continue

        # Handle potential multi-channel masks (take first channel)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mapped = remap_mask(mask)

        unique, counts = np.unique(mapped, return_counts=True)
        for val, count in zip(unique, counts):
            if val < num_classes:  # skip ignore index (255)
                pixel_counts[val] += count

    total_pixels = np.sum(pixel_counts)
    if total_pixels == 0:
        raise ValueError("No valid pixels found. Check your mask directory path.")

    class_frequencies = pixel_counts / total_pixels

    # Inverse frequency weighting with epsilon to avoid div-by-zero
    epsilon = 1e-6
    weights = 1.0 / (class_frequencies + epsilon)
    weights = weights / np.sum(weights) * num_classes  # normalize

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Class':<5} {'Class Name':<20} {'Pixel %':>8} {'Weight':>10}")
    print("=" * 55)
    for i in range(num_classes):
        marker = " <-- RARE" if class_frequencies[i] < 0.01 else ""
        print(f"{i:<5} {CLASS_NAMES[i]:<20} "
              f"{class_frequencies[i]*100:>7.3f}% "
              f"{weights[i]:>10.4f}{marker}")
    print("=" * 55)
    print(f"Total pixels analyzed: {int(total_pixels):,}\n")

    return torch.FloatTensor(weights)


class EpochLogger:
    """Prints a clean per-class IoU table to the terminal after each epoch."""

    def __init__(self, class_names=CLASS_NAMES):
        self.class_names = class_names
        self.history = []

    def log_epoch(self, epoch, train_loss, val_loss, val_iou_list, mIoU):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | mIoU: {mIoU:.4f}")
        print("-" * 50)
        print(f"{'Class Name':<20} | {'Val IoU':>10}")
        print("-" * 50)

        for name, iou in zip(self.class_names, val_iou_list):
            if np.isnan(iou):
                iou_str = "   N/A"
                flag = "  (no samples)"
            elif iou < 0.30:
                iou_str = f"{iou:.4f}"
                flag = "  !! LOW"
            else:
                iou_str = f"{iou:.4f}"
                flag = ""
            print(f"{name:<20} | {iou_str:>10}{flag}")

        print("-" * 50)

        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mIoU': mIoU,
            'per_class_iou': val_iou_list
        })

    def get_best_epoch(self):
        if not self.history:
            return None
        return max(self.history, key=lambda x: x['mIoU'])


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Update this path to your actual Segmentation mask folder
    # Based on your directory structure:
    # C:\Users\drona\Downloads\Offroad_Segmentation_Scripts\train\Segmentation
    # ----------------------------------------------------------------
    import sys

    mask_dir = os.path.join(
        r"C:\Users\drona\Downloads\Offroad_Segmentation_Scripts",
        "train", "Segmentation"
    )

    if not os.path.exists(mask_dir):
        print(f"ERROR: Path not found: {mask_dir}")
        print("Please update the mask_dir path in this script.")
        sys.exit(1)

    weights = analyze_class_frequencies(mask_dir)
    print(f"Loss weights tensor: {weights}")