"""
Segmentation Data Utilities
============================
Image/mask loading, patch extraction, augmentation, and dataset classes
for the UMOD urine sediment segmentation pipeline.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

# UMOD dataset class mapping (0-7)
CLASS_NAMES = {
    0: 'Background',
    1: 'RBC',           # Red Blood Cells
    2: 'WBC',           # White Blood Cells
    3: 'Bacteria',
    4: 'Small EPC',     # Small Epithelial Cells
    5: 'Large EPC',     # Large Epithelial Cells
    6: 'EPC Sheet',     # Epithelial Cell Sheets
    7: 'Yeast',
}

NUM_CLASSES = 8  # 0 = background + 7 cell types

# ImageNet normalization stats (for pretrained encoders)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image dimensions
ORIGINAL_H, ORIGINAL_W = 1040, 1392
PATCH_SIZE = 512


def load_image(path: str) -> np.ndarray:
    """Load a .tif image as RGB numpy array (H, W, 3), uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask(path: str) -> np.ndarray:
    """Load a multi-class mask as numpy array (H, W), values 0-7."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    # Clip to valid range
    mask = np.clip(mask, 0, NUM_CLASSES - 1)
    return mask


def load_split(data_root: str, split: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load all images and masks for a given split.

    Parameters
    ----------
    data_root : str
        Root path to ds1/ds1/
    split : str
        'train', 'validation', or 'test'

    Returns
    -------
    images : list of np.ndarray (H, W, 3)
    masks : list of np.ndarray (H, W)
    filenames : list of str
    """
    img_dir = Path(data_root) / split / "img" / "cls"
    mask_dir = Path(data_root) / split / "mult_mask" / "cls"

    filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])

    images = []
    masks = []
    valid_filenames = []

    for fname in filenames:
        img = load_image(img_dir / fname)
        msk = load_mask(mask_dir / fname)
        images.append(img)
        masks.append(msk)
        valid_filenames.append(fname)

    return images, masks, valid_filenames


def compute_class_statistics(masks: List[np.ndarray]) -> Dict:
    """
    Compute per-class pixel and object statistics across all masks.

    Returns dict with per-class: total_pixels, mean_pixels, n_images_present,
    object_count (from connected components).
    """
    stats = {}
    for cls_id in range(NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_id]
        pixel_counts = []
        object_counts = []

        for mask in masks:
            binary = (mask == cls_id).astype(np.uint8)
            px_count = binary.sum()
            pixel_counts.append(int(px_count))

            if px_count > 0:
                n_objects, _ = cv2.connectedComponents(binary)
                object_counts.append(n_objects - 1)  # subtract background component
            else:
                object_counts.append(0)

        stats[cls_name] = {
            'class_id': cls_id,
            'total_pixels': int(np.sum(pixel_counts)),
            'mean_pixels_per_image': float(np.mean(pixel_counts)),
            'n_images_present': int(np.sum(np.array(pixel_counts) > 0)),
            'total_objects': int(np.sum(object_counts)),
            'mean_objects_per_image': float(np.mean(object_counts)),
        }

    return stats


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = 256,
    min_foreground_ratio: float = 0.005,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract patches from an image/mask pair.

    Parameters
    ----------
    image : np.ndarray (H, W, 3)
    mask : np.ndarray (H, W)
    patch_size : int
    stride : int
    min_foreground_ratio : float
        Minimum foreground pixel ratio to keep a patch (filters >99.5% background)

    Returns
    -------
    list of (image_patch, mask_patch) tuples
    """
    h, w = image.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y:y + patch_size, x:x + patch_size]
            msk_patch = mask[y:y + patch_size, x:x + patch_size]

            # Check foreground ratio
            fg_ratio = (msk_patch > 0).sum() / (patch_size * patch_size)
            if fg_ratio >= min_foreground_ratio:
                patches.append((img_patch, msk_patch))

    return patches


def pad_to_inference_size(image: np.ndarray, target_h: int = 1536, target_w: int = 1536) -> np.ndarray:
    """Pad image to target size with zero padding (for inference stitching)."""
    h, w = image.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    if len(image.shape) == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    return padded


def stitch_patches(patches: List[np.ndarray], grid_shape: Tuple[int, int],
                   patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Stitch non-overlapping patches back into a full image.

    Parameters
    ----------
    patches : list of np.ndarray, each (patch_size, patch_size) or (patch_size, patch_size, C)
    grid_shape : (n_rows, n_cols)
    patch_size : int

    Returns
    -------
    np.ndarray : stitched image
    """
    n_rows, n_cols = grid_shape
    if len(patches[0].shape) == 3:
        c = patches[0].shape[2]
        full = np.zeros((n_rows * patch_size, n_cols * patch_size, c), dtype=patches[0].dtype)
    else:
        full = np.zeros((n_rows * patch_size, n_cols * patch_size), dtype=patches[0].dtype)

    idx = 0
    for r in range(n_rows):
        for c_idx in range(n_cols):
            y = r * patch_size
            x = c_idx * patch_size
            full[y:y + patch_size, x:x + patch_size] = patches[idx]
            idx += 1

    return full


def get_inference_patches(image: np.ndarray, patch_size: int = PATCH_SIZE) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """
    Pad image and extract non-overlapping patches for inference.

    Returns patches and grid_shape for stitching.
    """
    # Pad to nearest multiple of patch_size
    h, w = image.shape[:2]
    target_h = ((h + patch_size - 1) // patch_size) * patch_size
    target_w = ((w + patch_size - 1) // patch_size) * patch_size

    padded = pad_to_inference_size(image, target_h, target_w)

    n_rows = target_h // patch_size
    n_cols = target_w // patch_size

    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            y = r * patch_size
            x = c * patch_size
            patches.append(padded[y:y + patch_size, x:x + patch_size])

    return patches, (n_rows, n_cols)


def get_train_augmentation(patch_size: int = PATCH_SIZE) -> 'A.Compose':
    """Get training augmentation pipeline."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for augmentation")

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ElasticTransform(alpha=50, sigma=10, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_augmentation(patch_size: int = PATCH_SIZE) -> 'A.Compose':
    """Get validation/test augmentation pipeline (normalize only)."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for augmentation")

    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class SegmentationPatchDataset(Dataset):
    """PyTorch dataset for segmentation patches."""

    def __init__(self, patches: List[Tuple[np.ndarray, np.ndarray]],
                 transform=None):
        """
        Parameters
        ----------
        patches : list of (image_patch, mask_patch) tuples
        transform : albumentations.Compose or None
        """
        self.patches = patches
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image, mask = self.patches[idx]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # (C, H, W) float tensor
            mask = augmented['mask']    # (H, W) long tensor
        else:
            # Manual conversion
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask = torch.from_numpy(mask)

        mask = mask.long()
        return image, mask


class FullImageDataset(Dataset):
    """PyTorch dataset for full-resolution inference."""

    def __init__(self, images: List[np.ndarray], masks: List[np.ndarray] = None,
                 filenames: List[str] = None, transform=None):
        self.images = images
        self.masks = masks
        self.filenames = filenames or [f"img_{i}" for i in range(len(images))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx] if self.masks is not None else None

        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask'].long()
            else:
                augmented = self.transform(image=image)
                image = augmented['image']

        return image, mask, self.filenames[idx]


def compute_class_weights(masks: List[np.ndarray], fg_cap: float = 10.0) -> np.ndarray:
    """
    Compute class weights for background-dominated segmentation.

    Strategy:
    - Background weight = 1.0 (never downweight background)
    - Foreground weights = sqrt(inverse_frequency), capped and normalized
      so mean foreground weight = 1.0

    This avoids the pathology where aggressive inverse-frequency weighting
    causes models to over-predict foreground.

    Parameters
    ----------
    masks : list of np.ndarray
    fg_cap : float
        Maximum weight cap for foreground classes

    Returns
    -------
    np.ndarray of shape (NUM_CLASSES,)
    """
    total_pixels = np.zeros(NUM_CLASSES, dtype=np.float64)

    for mask in masks:
        for cls_id in range(NUM_CLASSES):
            total_pixels[cls_id] += (mask == cls_id).sum()

    weights = np.ones(NUM_CLASSES, dtype=np.float64)

    # Background weight stays at 1.0
    # Foreground: sqrt of inverse frequency (among foreground only)
    fg_total = total_pixels[1:].sum()
    fg_freq = total_pixels[1:] / (fg_total + 1e-8)
    median_freq = np.median(fg_freq[fg_freq > 0])
    fg_weights = median_freq / (fg_freq + 1e-8)
    fg_weights = np.sqrt(fg_weights)
    fg_weights = np.minimum(fg_weights, fg_cap)
    fg_weights = fg_weights / (fg_weights.mean() + 1e-8)  # mean fg weight = 1.0

    weights[1:] = fg_weights

    return weights.astype(np.float32)
