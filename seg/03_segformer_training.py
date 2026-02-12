"""
03_segformer_training.py
========================
SegFormer-B2 for 8-class Urine Sediment Segmentation

Architecture: SegformerForSemanticSegmentation from HuggingFace (nvidia/segformer-b2-finetuned-ade-512-512)
Loss: Same CompoundLoss as U-Net (fair comparison)
Training: 200 epochs, batch_size=4, AdamW lr=6e-5, PolynomialLR with 5-epoch warmup

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import SegformerForSemanticSegmentation, SegformerConfig

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.seg_data_utils import (
    SegmentationPatchDataset, get_train_augmentation, get_val_augmentation,
    NUM_CLASSES, PATCH_SIZE
)
from utils.seg_metrics import mean_dice, binary_dice
from seg.models.losses import CompoundLoss

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "seg_models"

# Training hyperparameters
BATCH_SIZE = 4  # Smaller batch for larger model
NUM_EPOCHS = 200
LR = 6e-5
WARMUP_EPOCHS = 5
PATIENCE = 30


class SegFormerWrapper(nn.Module):
    """
    Wrapper around HuggingFace SegFormer to output logits at input resolution.

    SegFormer outputs at 1/4 resolution, so we upsample to match target masks.
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) float tensor

        Returns
        -------
        logits : (B, num_classes, H, W) float tensor at input resolution
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits


def get_polynomial_lr_lambda(epoch, total_epochs, warmup_epochs, power=0.9):
    """Polynomial LR decay with linear warmup."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return (1 - progress) ** power


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    dice_scores = []
    binary_dice_scores = []
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        masks_np = masks.cpu().numpy()

        for pred, gt in zip(preds, masks_np):
            dice_scores.append(mean_dice(pred, gt, NUM_CLASSES))
            binary_dice_scores.append(binary_dice(pred, gt))

    avg_loss = total_loss / n_batches
    avg_dice = float(np.mean(dice_scores))
    avg_binary_dice = float(np.mean(binary_dice_scores))

    return avg_loss, avg_dice, avg_binary_dice


def plot_training_curves(history: dict, save_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train', color='#3498db', lw=2)
    axes[0].plot(epochs, history['val_loss'], label='Val', color='#e74c3c', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_dice'], label='Val Mean Dice', color='#2ecc71', lw=2)
    axes[1].axhline(y=max(history['val_dice']), color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Dice')
    axes[1].set_title(f'Validation Mean Dice (best: {max(history["val_dice"]):.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['val_binary_dice'], label='Val Binary Dice', color='#9b59b6', lw=2)
    axes[2].axhline(y=0.877, color='red', linestyle='--', label='Paper Baseline (0.877)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Binary Dice')
    axes[2].set_title('Binary Dice (Foreground vs Background)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('SegFormer-B2 Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("SEGFORMER-B2 TRAINING")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Load patches
    print("\n[Step 1] Loading patches...")
    train_imgs = np.load(RESULTS_DIR / 'patches_train_images.npy')
    train_masks = np.load(RESULTS_DIR / 'patches_train_masks.npy')
    val_imgs = np.load(RESULTS_DIR / 'patches_validation_images.npy')
    val_masks = np.load(RESULTS_DIR / 'patches_validation_masks.npy')

    train_patches = list(zip(train_imgs, train_masks))
    val_patches = list(zip(val_imgs, val_masks))

    # Step 2: Create datasets
    print("\n[Step 2] Creating datasets...")
    train_dataset = SegmentationPatchDataset(train_patches, transform=get_train_augmentation())
    val_dataset = SegmentationPatchDataset(val_patches, transform=get_val_augmentation())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Step 3: Build model
    print("\n[Step 3] Building SegFormer-B2...")
    model = SegFormerWrapper(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Step 4: Setup training
    print("\n[Step 4] Setting up training...")
    class_weights = np.load(RESULTS_DIR / 'class_weights.npy')
    criterion = CompoundLoss(num_classes=NUM_CLASSES, class_weights=class_weights)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: get_polynomial_lr_lambda(e, NUM_EPOCHS, WARMUP_EPOCHS)
    )

    # Step 5: Training loop
    print(f"\n[Step 5] Training for {NUM_EPOCHS} epochs...")
    history = {
        'train_loss': [], 'val_loss': [],
        'val_dice': [], 'val_binary_dice': [],
    }

    best_dice = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()

        val_loss, val_dice, val_binary_dice = validate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_binary_dice'].append(val_binary_dice)

        if epoch % 10 == 0 or epoch <= 5:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Dice={val_dice:.4f}, LR={lr:.2e}")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_binary_dice': val_binary_dice,
            }, MODELS_DIR / 'segformer_best.pth')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time

    # Step 6: Save results
    print("\n[Step 6] Saving results...")
    plot_training_curves(history, FIGURES_DIR / 'fig05_segformer_training_curves.png')

    results = {
        'model': 'SegFormer-B2',
        'pretrained': 'nvidia/segformer-b2-finetuned-ade-512-512',
        'num_classes': NUM_CLASSES,
        'patch_size': PATCH_SIZE,
        'batch_size': BATCH_SIZE,
        'num_epochs': len(history['train_loss']),
        'best_val_dice': float(best_dice),
        'best_val_binary_dice': float(max(history['val_binary_dice'])),
        'training_time_seconds': training_time,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'hyperparameters': {
            'learning_rate': LR,
            'warmup_epochs': WARMUP_EPOCHS,
            'weight_decay': 1e-4,
            'scheduler': 'PolynomialLR (power=0.9) with linear warmup',
            'loss': 'CompoundLoss (0.5 Dice + 0.5 Focal)',
            'patience': PATIENCE,
        }
    }

    with open(RESULTS_DIR / 'segformer_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SEGFORMER-B2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val Mean Dice: {best_dice:.4f}")
    print(f"  Best Val Binary Dice: {max(history['val_binary_dice']):.4f}")
    print(f"  Training time: {training_time / 60:.1f} min")


if __name__ == "__main__":
    main()
