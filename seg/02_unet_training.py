"""
02_unet_training.py
====================
U-Net with ResNet34 Encoder for 8-class Urine Sediment Segmentation

Architecture: segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=8)
Loss: Compound (0.5 Dice + 0.5 Focal), class weights from inverse frequency
Training: 200 epochs, batch_size=8, AdamW lr=1e-4, CosineAnnealingWarmRestarts
Encoder frozen first 10 epochs, then unfrozen at 0.1x LR

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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import segmentation_models_pytorch as smp

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
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 200
LR = 1e-4
ENCODER_UNFREEZE_EPOCH = 10
ENCODER_LR_MULT = 0.1
PATIENCE = 30


def build_model() -> nn.Module:
    """Build U-Net with ResNet34 encoder pretrained on ImageNet."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,  # Raw logits
    )
    return model


def freeze_encoder(model: nn.Module):
    """Freeze encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module):
    """Unfreeze encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = True


def get_optimizer(model: nn.Module, lr: float, encoder_unfrozen: bool) -> AdamW:
    """Get optimizer with optional differential learning rates."""
    if encoder_unfrozen:
        encoder_params = list(model.encoder.parameters())
        decoder_params = [p for name, p in model.named_parameters()
                          if not name.startswith('encoder')]
        param_groups = [
            {'params': encoder_params, 'lr': lr * ENCODER_LR_MULT},
            {'params': decoder_params, 'lr': lr},
        ]
    else:
        decoder_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{'params': decoder_params, 'lr': lr}]

    return AdamW(param_groups, weight_decay=1e-4)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
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
    """Validate model, return average loss and mean Dice."""
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

        # Compute Dice on batch
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

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#3498db', lw=2)
    axes[0].plot(epochs, history['val_loss'], label='Val', color='#e74c3c', lw=2)
    axes[0].axvline(x=ENCODER_UNFREEZE_EPOCH, color='gray', linestyle=':', label='Encoder Unfreeze')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mean Dice
    axes[1].plot(epochs, history['val_dice'], label='Val Mean Dice', color='#2ecc71', lw=2)
    axes[1].axhline(y=max(history['val_dice']), color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Dice')
    axes[1].set_title(f'Validation Mean Dice (best: {max(history["val_dice"]):.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Binary Dice
    axes[2].plot(epochs, history['val_binary_dice'], label='Val Binary Dice', color='#9b59b6', lw=2)
    axes[2].axhline(y=0.877, color='red', linestyle='--', label='Paper Baseline (0.877)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Binary Dice')
    axes[2].set_title('Binary Dice (Foreground vs Background)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('U-Net (ResNet34) Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("U-NET (ResNet34) TRAINING")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Load patches
    print("\n[Step 1] Loading patches...")
    train_imgs = np.load(RESULTS_DIR / 'patches_train_images.npy')
    train_masks = np.load(RESULTS_DIR / 'patches_train_masks.npy')
    val_imgs = np.load(RESULTS_DIR / 'patches_validation_images.npy')
    val_masks = np.load(RESULTS_DIR / 'patches_validation_masks.npy')

    print(f"  Train patches: {train_imgs.shape}")
    print(f"  Val patches: {val_imgs.shape}")

    # Convert to patch lists
    train_patches = list(zip(train_imgs, train_masks))
    val_patches = list(zip(val_imgs, val_masks))

    # Step 2: Create datasets and loaders
    print("\n[Step 2] Creating datasets...")
    train_dataset = SegmentationPatchDataset(train_patches, transform=get_train_augmentation())
    val_dataset = SegmentationPatchDataset(val_patches, transform=get_val_augmentation())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Step 3: Build model
    print("\n[Step 3] Building model...")
    model = build_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Step 4: Setup training
    print("\n[Step 4] Setting up training...")
    class_weights = np.load(RESULTS_DIR / 'class_weights.npy')
    criterion = CompoundLoss(num_classes=NUM_CLASSES, class_weights=class_weights)

    # Freeze encoder initially
    freeze_encoder(model)
    optimizer = get_optimizer(model, LR, encoder_unfrozen=False)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Step 5: Training loop
    print(f"\n[Step 5] Training for {NUM_EPOCHS} epochs...")
    history = {
        'train_loss': [], 'val_loss': [],
        'val_dice': [], 'val_binary_dice': [],
    }

    best_dice = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Unfreeze encoder after ENCODER_UNFREEZE_EPOCH
        if epoch == ENCODER_UNFREEZE_EPOCH + 1:
            print(f"\n  >>> Unfreezing encoder at epoch {epoch} <<<")
            unfreeze_encoder(model)
            optimizer = get_optimizer(model, LR, encoder_unfrozen=True)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()

        # Validate
        val_loss, val_dice, val_binary_dice = validate(model, val_loader, criterion, DEVICE)

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_binary_dice'].append(val_binary_dice)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Dice={val_dice:.4f}, Binary Dice={val_binary_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_binary_dice': val_binary_dice,
            }, MODELS_DIR / 'unet_best.pth')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    training_time = time.time() - start_time

    # Step 6: Save results
    print("\n[Step 6] Saving results...")
    plot_training_curves(history, FIGURES_DIR / 'fig04_unet_training_curves.png')

    results = {
        'model': 'U-Net (ResNet34)',
        'encoder': 'resnet34',
        'pretrained': 'imagenet',
        'num_classes': NUM_CLASSES,
        'patch_size': PATCH_SIZE,
        'batch_size': BATCH_SIZE,
        'num_epochs': len(history['train_loss']),
        'best_val_dice': float(best_dice),
        'best_val_binary_dice': float(max(history['val_binary_dice'])),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'training_time_seconds': training_time,
        'total_parameters': total_params,
        'hyperparameters': {
            'learning_rate': LR,
            'encoder_lr_multiplier': ENCODER_LR_MULT,
            'encoder_unfreeze_epoch': ENCODER_UNFREEZE_EPOCH,
            'weight_decay': 1e-4,
            'scheduler': 'CosineAnnealingWarmRestarts',
            'loss': 'CompoundLoss (0.5 Dice + 0.5 Focal)',
            'patience': PATIENCE,
        }
    }

    with open(RESULTS_DIR / 'unet_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("U-NET TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val Mean Dice: {best_dice:.4f}")
    print(f"  Best Val Binary Dice: {max(history['val_binary_dice']):.4f}")
    print(f"  Paper Baseline Binary Dice: 0.877")
    print(f"  Training time: {training_time / 60:.1f} min")


if __name__ == "__main__":
    main()
