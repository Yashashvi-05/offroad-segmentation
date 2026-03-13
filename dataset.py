import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Complete 10-class mapping (600=Flowers was missing in original)
VALUE_MAP = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9   # Sky
}

NUM_CLASSES = 10
IMG_HEIGHT = 512
IMG_WIDTH = 512


def remap_mask(mask):
    """
    Safely remaps raw mask pixel values to class indices 0-9.
    Pixels not in VALUE_MAP get assigned 255 (ignore index for CrossEntropyLoss).
    This handles large values like 7100 and 10000 safely without giant arrays.
    """
    output = np.full(mask.shape, 255, dtype=np.uint8)
    for raw_val, class_idx in VALUE_MAP.items():
        output[mask == raw_val] = class_idx
    return output


def get_train_transforms(height=IMG_HEIGHT, width=IMG_WIDTH):
    """
    Domain-shift-focused augmentation pipeline.
    RandomShadow simulates time-of-day lighting variation.
    CoarseDropout simulates occlusion (especially helps Logs class).
    All transforms are applied consistently to both image and mask.
    """
    return A.Compose([
        A.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.3
        ),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.2
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(height=IMG_HEIGHT, width=IMG_WIDTH):
    """Minimal transforms for validation — only resize and normalize."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


class OffroadSegmentationDataset(Dataset):
    """
    Dataset loader for the Duality AI Offroad Segmentation challenge.

    Expected folder structure (based on your extracted directory):
        <split_dir>/
            Color_Images/    <- RGB .png files
            Segmentation/    <- Mask .png files (same filenames as Color_Images)

    Args:
        split_dir: Full path to 'train' or 'val' folder
        is_train: If True, applies domain-shift augmentations.
                  If False, applies only resize + normalize.
    """

    def __init__(self, split_dir, is_train=True):
        self.image_dir = os.path.join(split_dir, 'Color_Images')
        self.mask_dir = os.path.join(split_dir, 'Segmentation')
        self.is_train = is_train

        # Validate paths exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                f"Color_Images folder not found at: {self.image_dir}\n"
                f"Check that split_dir points to your 'train' or 'val' folder."
            )
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(
                f"Segmentation folder not found at: {self.mask_dir}"
            )

        # Get filenames from image dir, filter to PNG only
        all_files = os.listdir(self.image_dir)
        self.data_ids = [f for f in all_files if f.lower().endswith('.png')]

        if not self.data_ids:
            raise ValueError(f"No PNG images found in {self.image_dir}")

        self.transform = (get_train_transforms() if is_train
                         else get_val_transforms())

        print(f"{'Train' if is_train else 'Val'} dataset loaded: "
              f"{len(self.data_ids)} images from {split_dir}")

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        filename = self.data_ids[idx]

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Load image (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as unchanged to preserve raw class ID values
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError(f"Could not read mask: {mask_path}")

        # Handle multi-channel masks — take first channel only
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Remap raw pixel values to class indices 0-9
        mask = remap_mask(mask)

        # Apply augmentations (albumentations handles image+mask together)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']          # (C, H, W) float tensor
        mask = transformed['mask'].long()     # (H, W) long tensor

        return image, mask


class OffroadTestDataset(Dataset):
    """
    Dataset loader for test images (no masks — unseen environment).
    Used with test_segformer.py to generate predictions.

    Args:
        test_dir: Full path to folder containing test RGB images
    """

    def __init__(self, test_dir):
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        all_files = os.listdir(test_dir)
        self.image_paths = [
            os.path.join(test_dir, f)
            for f in all_files
            if f.lower().endswith('.png')
        ]
        self.filenames = [f for f in all_files if f.lower().endswith('.png')]

        if not self.image_paths:
            raise ValueError(f"No PNG images found in {test_dir}")

        self.transform = get_val_transforms()
        print(f"Test dataset loaded: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image = transformed['image']

        return image, self.filenames[idx]


if __name__ == "__main__":
    # Quick smoke test — update this path to your actual train folder
    import sys

    train_dir = r"C:\Users\drona\Downloads\Offroad_Segmentation_Scripts\train"

    if not os.path.exists(train_dir):
        print(f"ERROR: Path not found: {train_dir}")
        sys.exit(1)

    dataset = OffroadSegmentationDataset(train_dir, is_train=True)
    image, mask = dataset[0]

    print(f"Image shape: {image.shape}")   # expect (3, 512, 512)
    print(f"Mask shape:  {mask.shape}")    # expect (512, 512)
    print(f"Mask unique values: {torch.unique(mask).tolist()}")  # expect 0-9
    print(f"Image dtype: {image.dtype}")   # expect torch.float32
    print(f"Mask dtype:  {mask.dtype}")    # expect torch.int64
    print("Smoke test passed.")