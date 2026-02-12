"""
04_seg_evaluation.py
=====================
Segmentation Model Evaluation

Evaluates both U-Net and SegFormer on the test set:
1. Per-class Dice, IoU, Precision, Recall (on full 1392x1040 images)
2. Binary Dice for comparison with paper baseline (0.877)
3. Statistical comparison: paired Wilcoxon test + bootstrap CI
4. Qualitative comparison grid and confusion matrices

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.seg_data_utils import (
    load_split, get_inference_patches, stitch_patches,
    get_val_augmentation, CLASS_NAMES, NUM_CLASSES, PATCH_SIZE, ORIGINAL_H, ORIGINAL_W
)
from utils.seg_metrics import (
    per_class_metrics, mean_dice, binary_dice, pixel_accuracy,
    confusion_matrix_seg, aggregate_metrics
)

import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SegFormerWrapper(nn.Module):
    """Wrapper to upsample SegFormer output to input resolution."""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits
DATA_ROOT = Path("c:/Users/SLKhurram/Downloads/Images/ds1/ds1")
RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "seg_models"


def load_model(model_name: str) -> torch.nn.Module:
    """Load a trained segmentation model."""
    if model_name == 'unet':
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=NUM_CLASSES, activation=None)
        checkpoint = torch.load(MODELS_DIR / 'unet_best.pth', map_location=DEVICE)
    elif model_name == 'segformer':
        model = SegFormerWrapper(num_classes=NUM_CLASSES)
        checkpoint = torch.load(MODELS_DIR / 'segformer_best.pth', map_location=DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_full_image(model: torch.nn.Module, image: np.ndarray,
                       transform=None) -> np.ndarray:
    """
    Predict segmentation mask for a full-resolution image using patch stitching.

    1. Pad to nearest multiple of PATCH_SIZE
    2. Extract non-overlapping patches
    3. Predict each patch
    4. Stitch back and crop to original size

    Returns
    -------
    np.ndarray (H, W) predicted mask
    """
    patches, grid_shape = get_inference_patches(image, PATCH_SIZE)

    predicted_patches = []
    for patch in patches:
        if transform is not None:
            augmented = transform(image=patch)
            patch_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        else:
            patch_tensor = torch.from_numpy(
                patch.transpose(2, 0, 1).astype(np.float32) / 255.0
            ).unsqueeze(0).to(DEVICE)

        logits = model(patch_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        predicted_patches.append(pred)

    # Stitch
    full_pred = stitch_patches(predicted_patches, grid_shape, PATCH_SIZE)

    # Crop to original size
    full_pred = full_pred[:ORIGINAL_H, :ORIGINAL_W]

    return full_pred


def evaluate_model_on_split(model, images, masks, transform, model_name: str):
    """Evaluate a model on a set of images, return per-image metrics."""
    all_metrics = []
    all_dice = []
    all_binary_dice = []
    all_predictions = []

    for i, (img, gt) in enumerate(zip(images, masks)):
        pred = predict_full_image(model, img, transform)
        all_predictions.append(pred)

        metrics = per_class_metrics(pred, gt, NUM_CLASSES)
        all_metrics.append(metrics)

        md = mean_dice(pred, gt, NUM_CLASSES)
        bd = binary_dice(pred, gt)
        pa = pixel_accuracy(pred, gt)

        all_dice.append(md)
        all_binary_dice.append(bd)

        if (i + 1) % 20 == 0:
            print(f"  [{model_name}] {i+1}/{len(images)}: "
                  f"Mean Dice={md:.4f}, Binary Dice={bd:.4f}")

    return all_metrics, all_dice, all_binary_dice, all_predictions


def plot_dice_comparison(unet_dice, segformer_dice, save_path: Path):
    """Bar chart comparing per-class Dice between models."""
    unet_summary = aggregate_metrics(unet_dice, NUM_CLASSES)
    sf_summary = aggregate_metrics(segformer_dice, NUM_CLASSES)

    classes = [CLASS_NAMES[i] for i in range(1, NUM_CLASSES)]
    unet_means = [unet_summary[i]['dice']['mean'] for i in range(1, NUM_CLASSES)]
    sf_means = [sf_summary[i]['dice']['mean'] for i in range(1, NUM_CLASSES)]
    unet_stds = [unet_summary[i]['dice']['std'] for i in range(1, NUM_CLASSES)]
    sf_stds = [sf_summary[i]['dice']['std'] for i in range(1, NUM_CLASSES)]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, unet_means, width, yerr=unet_stds, capsize=3,
                   label='U-Net (ResNet34)', color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, sf_means, width, yerr=sf_stds, capsize=3,
                   label='SegFormer-B2', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Dice Score')
    ax.set_title('Per-Class Dice Score Comparison (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_qualitative_comparison(images, masks, unet_preds, sf_preds,
                                filenames, save_path: Path, n_samples: int = 4):
    """Grid showing image, GT, U-Net pred, SegFormer pred."""
    indices = np.linspace(0, len(images) - 1, n_samples, dtype=int)

    cmap_colors = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 0],
    ], dtype=np.uint8)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    for row, idx in enumerate(indices):
        axes[row, 0].imshow(images[idx])
        axes[row, 0].set_title(f'Image ({filenames[idx]})', fontsize=9)

        axes[row, 1].imshow(cmap_colors[masks[idx]])
        axes[row, 1].set_title('Ground Truth', fontsize=9)

        axes[row, 2].imshow(cmap_colors[unet_preds[idx]])
        axes[row, 2].set_title('U-Net Prediction', fontsize=9)

        axes[row, 3].imshow(cmap_colors[sf_preds[idx]])
        axes[row, 3].set_title('SegFormer Prediction', fontsize=9)

        for col in range(4):
            axes[row, col].axis('off')

    plt.suptitle('Qualitative Segmentation Comparison (Test Set)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(unet_preds, sf_preds, gt_masks, save_path: Path):
    """Plot pixel-level confusion matrices for both models."""
    unet_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    sf_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for pred, gt in zip(unet_preds, gt_masks):
        unet_cm += confusion_matrix_seg(pred, gt, NUM_CLASSES)
    for pred, gt in zip(sf_preds, gt_masks):
        sf_cm += confusion_matrix_seg(pred, gt, NUM_CLASSES)

    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, cm, title in zip(axes, [unet_cm, sf_cm], ['U-Net', 'SegFormer-B2']):
        # Normalize rows
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=ax, vmin=0, vmax=1, annot_kws={'size': 7})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{title} Pixel-Level Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("SEGMENTATION MODEL EVALUATION")
    print("=" * 60)

    # Step 1: Load test data
    print("\n[Step 1] Loading test images...")
    test_images, test_masks, test_filenames = load_split(str(DATA_ROOT), 'test')
    print(f"  Loaded {len(test_images)} test images")

    transform = get_val_augmentation()

    # Step 2: Evaluate U-Net
    print("\n[Step 2] Evaluating U-Net...")
    unet_model = load_model('unet')
    unet_metrics, unet_dice_list, unet_bdice_list, unet_preds = \
        evaluate_model_on_split(unet_model, test_images, test_masks, transform, 'U-Net')

    print(f"  U-Net Mean Dice: {np.mean(unet_dice_list):.4f} +/- {np.std(unet_dice_list):.4f}")
    print(f"  U-Net Binary Dice: {np.mean(unet_bdice_list):.4f} +/- {np.std(unet_bdice_list):.4f}")

    # Step 3: Evaluate SegFormer
    print("\n[Step 3] Evaluating SegFormer-B2...")
    sf_model = load_model('segformer')
    sf_metrics, sf_dice_list, sf_bdice_list, sf_preds = \
        evaluate_model_on_split(sf_model, test_images, test_masks, transform, 'SegFormer')

    print(f"  SegFormer Mean Dice: {np.mean(sf_dice_list):.4f} +/- {np.std(sf_dice_list):.4f}")
    print(f"  SegFormer Binary Dice: {np.mean(sf_bdice_list):.4f} +/- {np.std(sf_bdice_list):.4f}")

    # Step 4: Statistical comparison
    print("\n[Step 4] Statistical comparison...")
    wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(unet_dice_list, sf_dice_list)
    print(f"  Paired Wilcoxon test (Mean Dice): stat={wilcoxon_stat:.4f}, p={wilcoxon_p:.6f}")

    diff = np.array(unet_dice_list) - np.array(sf_dice_list)
    bootstrap_diffs = []
    for _ in range(1000):
        idx = np.random.choice(len(diff), len(diff), replace=True)
        bootstrap_diffs.append(diff[idx].mean())
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    print(f"  Dice difference (U-Net - SegFormer): {diff.mean():.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Step 5: Generate outputs
    print("\n[Step 5] Generating visualizations and tables...")

    # Save predictions for feature extraction
    np.save(RESULTS_DIR / 'test_predictions_unet.npy', np.array(unet_preds, dtype=np.uint8))
    np.save(RESULTS_DIR / 'test_predictions_segformer.npy', np.array(sf_preds, dtype=np.uint8))

    # Also predict on train and val for feature extraction
    for split in ['train', 'validation']:
        print(f"\n  Predicting {split} set...")
        split_imgs, split_masks, split_fnames = load_split(str(DATA_ROOT), split)

        for model_name, model_obj in [('unet', unet_model), ('segformer', sf_model)]:
            preds = []
            for img in split_imgs:
                pred = predict_full_image(model_obj, img, transform)
                preds.append(pred)
            np.save(RESULTS_DIR / f'{split}_predictions_{model_name}.npy',
                    np.array(preds, dtype=np.uint8))

    # Dice comparison bar chart
    plot_dice_comparison(unet_metrics, sf_metrics, FIGURES_DIR / 'fig06_dice_comparison.png')

    # Qualitative comparison
    plot_qualitative_comparison(test_images, test_masks, unet_preds, sf_preds,
                                test_filenames, FIGURES_DIR / 'fig07_qualitative_comparison.png')

    # Confusion matrices
    plot_confusion_matrices(unet_preds, sf_preds, test_masks,
                            FIGURES_DIR / 'fig08_seg_confusion_matrices.png')

    # Build metrics CSV
    unet_agg = aggregate_metrics(unet_metrics, NUM_CLASSES)
    sf_agg = aggregate_metrics(sf_metrics, NUM_CLASSES)

    rows = []
    for cls_id in range(NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_id]
        for model_name, agg in [('U-Net', unet_agg), ('SegFormer-B2', sf_agg)]:
            rows.append({
                'Model': model_name,
                'Class': cls_name,
                'Dice': f"{agg[cls_id]['dice']['mean']:.4f} +/- {agg[cls_id]['dice']['std']:.4f}",
                'IoU': f"{agg[cls_id]['iou']['mean']:.4f} +/- {agg[cls_id]['iou']['std']:.4f}",
                'Precision': f"{agg[cls_id]['precision']['mean']:.4f}",
                'Recall': f"{agg[cls_id]['recall']['mean']:.4f}",
            })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / 'segmentation_metrics.csv', index=False)
    print(f"  Saved: {RESULTS_DIR / 'segmentation_metrics.csv'}")

    # Save complete results
    results = {
        'unet': {
            'mean_dice': float(np.mean(unet_dice_list)),
            'std_dice': float(np.std(unet_dice_list)),
            'mean_binary_dice': float(np.mean(unet_bdice_list)),
            'per_image_dice': [float(d) for d in unet_dice_list],
        },
        'segformer': {
            'mean_dice': float(np.mean(sf_dice_list)),
            'std_dice': float(np.std(sf_dice_list)),
            'mean_binary_dice': float(np.mean(sf_bdice_list)),
            'per_image_dice': [float(d) for d in sf_dice_list],
        },
        'comparison': {
            'wilcoxon_statistic': float(wilcoxon_stat),
            'wilcoxon_p_value': float(wilcoxon_p),
            'dice_diff_mean': float(diff.mean()),
            'dice_diff_ci_95': [float(ci_lower), float(ci_upper)],
        },
        'paper_baseline_binary_dice': 0.877,
    }

    with open(RESULTS_DIR / 'segmentation_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SEGMENTATION EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n  U-Net Mean Dice:      {np.mean(unet_dice_list):.4f}")
    print(f"  SegFormer Mean Dice:  {np.mean(sf_dice_list):.4f}")
    print(f"  Paper Baseline (bin): 0.877")
    print(f"  Wilcoxon p-value:     {wilcoxon_p:.6f}")


if __name__ == "__main__":
    main()
