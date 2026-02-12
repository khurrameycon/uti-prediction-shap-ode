"""
05_feature_extraction.py
=========================
Extract 36 Tabular Features from Segmentation Masks

Per-class (7 classes x 4 = 28 features):
  - {class}_count: number of connected components
  - {class}_total_area: total pixel area
  - {class}_mean_area: mean object area
  - {class}_max_area: largest object area

Cross-class (8 derived features):
  - total_cell_count, cell_density, foreground_ratio
  - bacteria_wbc_ratio, epithelial_total_area
  - epi_sheet_ratio, infection_signature, inflammation_score

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.seg_data_utils import load_split, CLASS_NAMES, NUM_CLASSES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOT = Path("c:/Users/SLKhurram/Downloads/Images/ds1/ds1")
RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"

# Feature class names (excluding background)
CELL_CLASSES = {1: 'rbc', 2: 'wbc', 3: 'bacteria', 4: 'small_epc',
                5: 'large_epc', 6: 'epc_sheet', 7: 'yeast'}


def extract_per_class_features(mask: np.ndarray) -> dict:
    """
    Extract 4 features per cell class from a single mask.

    Returns dict with keys like 'rbc_count', 'rbc_total_area', etc.
    """
    features = {}
    total_pixels = mask.size

    for cls_id, cls_name in CELL_CLASSES.items():
        binary = (mask == cls_id).astype(np.uint8)
        total_area = int(binary.sum())

        if total_area > 0:
            n_components, labels = cv2.connectedComponents(binary)
            n_objects = n_components - 1  # Subtract background

            # Object areas
            object_areas = []
            for obj_id in range(1, n_components):
                obj_area = (labels == obj_id).sum()
                object_areas.append(int(obj_area))

            mean_area = float(np.mean(object_areas)) if object_areas else 0.0
            max_area = float(np.max(object_areas)) if object_areas else 0.0
        else:
            n_objects = 0
            mean_area = 0.0
            max_area = 0.0

        features[f'{cls_name}_count'] = n_objects
        features[f'{cls_name}_total_area'] = total_area
        features[f'{cls_name}_mean_area'] = mean_area
        features[f'{cls_name}_max_area'] = max_area

    return features


def extract_cross_class_features(mask: np.ndarray, per_class: dict) -> dict:
    """
    Extract 8 cross-class derived features.
    """
    total_pixels = mask.size
    foreground = (mask > 0).sum()

    # Total cell count and density
    total_count = sum(per_class[f'{cn}_count'] for cn in CELL_CLASSES.values())
    cell_density = total_count / total_pixels if total_pixels > 0 else 0

    # Foreground ratio
    fg_ratio = foreground / total_pixels

    # Bacteria / WBC ratio (clinical marker)
    bacteria_area = per_class['bacteria_total_area']
    wbc_area = per_class['wbc_total_area']
    bacteria_wbc_ratio = bacteria_area / (wbc_area + 1) if wbc_area > 0 else bacteria_area

    # Epithelial features
    epi_total = (per_class['small_epc_total_area'] +
                 per_class['large_epc_total_area'] +
                 per_class['epc_sheet_total_area'])
    epi_sheet_ratio = per_class['epc_sheet_total_area'] / (epi_total + 1) if epi_total > 0 else 0

    # Infection signature: bacteria area / foreground area
    infection_sig = bacteria_area / (foreground + 1) if foreground > 0 else 0

    # Inflammation score: WBC area / foreground area
    inflammation = wbc_area / (foreground + 1) if foreground > 0 else 0

    return {
        'total_cell_count': total_count,
        'cell_density': cell_density,
        'foreground_ratio': fg_ratio,
        'bacteria_wbc_ratio': bacteria_wbc_ratio,
        'epithelial_total_area': epi_total,
        'epi_sheet_ratio': epi_sheet_ratio,
        'infection_signature': infection_sig,
        'inflammation_score': inflammation,
    }


def extract_all_features(mask: np.ndarray) -> dict:
    """Extract all 36 features from a single mask."""
    per_class = extract_per_class_features(mask)
    cross_class = extract_cross_class_features(mask, per_class)
    features = {**per_class, **cross_class}
    return features


def masks_to_feature_dataframe(masks: List, filenames: List[str], split: str) -> pd.DataFrame:
    """Convert a list of masks to a features DataFrame.

    Uses split-prefixed filenames (e.g. 'train/0001.tif') as index
    to avoid collisions since all splits share the same base filenames.
    """
    rows = []
    for mask, fname in zip(masks, filenames):
        features = extract_all_features(mask)
        features['filename'] = f"{split}/{fname}"
        rows.append(features)

    df = pd.DataFrame(rows)
    df = df.set_index('filename')
    return df


def log_transform_skewed(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """Apply log1p transform to highly skewed area features."""
    area_cols = [c for c in df.columns if 'area' in c or 'count' in c.lower()]
    for col in area_cols:
        skew = df[col].skew()
        if abs(skew) > threshold:
            df[col] = np.log1p(df[col])
    return df


def plot_feature_distributions(df_gt: pd.DataFrame, df_unet: pd.DataFrame,
                               df_sf: pd.DataFrame, save_path: Path):
    """Plot feature distributions comparing GT vs predicted features."""
    top_features = ['bacteria_count', 'wbc_count', 'rbc_count',
                    'infection_signature', 'inflammation_score', 'foreground_ratio']

    available = [f for f in top_features if f in df_gt.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        ax = axes[i]
        ax.hist(df_gt[feat], bins=20, alpha=0.5, label='GT', color='#2ecc71', edgecolor='black')
        ax.hist(df_unet[feat], bins=20, alpha=0.5, label='U-Net', color='#3498db', edgecolor='black')
        ax.hist(df_sf[feat], bins=20, alpha=0.5, label='SegFormer', color='#e74c3c', edgecolor='black')
        ax.set_title(feat.replace('_', ' ').title(), fontsize=10)
        ax.legend(fontsize=8)

    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distributions: GT vs Predicted Masks', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("FEATURE EXTRACTION FROM SEGMENTATION MASKS")
    print("=" * 60)

    # Step 1: Load GT masks and filenames for all splits
    print("\n[Step 1] Loading data...")
    all_gt_features = {}
    all_unet_features = {}
    all_sf_features = {}

    for split in ['train', 'validation', 'test']:
        print(f"\n  Processing {split}...")
        _, gt_masks, filenames = load_split(str(DATA_ROOT), split)

        # Load predicted masks
        unet_preds = np.load(RESULTS_DIR / f'{split}_predictions_unet.npy')
        sf_preds = np.load(RESULTS_DIR / f'{split}_predictions_segformer.npy')

        # Extract features from GT
        print(f"    Extracting GT features...")
        gt_df = masks_to_feature_dataframe(gt_masks, filenames, split)
        all_gt_features[split] = gt_df

        # Extract features from U-Net predictions
        print(f"    Extracting U-Net features...")
        unet_df = masks_to_feature_dataframe(list(unet_preds), filenames, split)
        all_unet_features[split] = unet_df

        # Extract features from SegFormer predictions
        print(f"    Extracting SegFormer features...")
        sf_df = masks_to_feature_dataframe(list(sf_preds), filenames, split)
        all_sf_features[split] = sf_df

        print(f"    {split}: {len(gt_df)} images, {len(gt_df.columns)} features")

    # Step 2: Combine splits
    print("\n[Step 2] Combining splits...")
    feature_names = list(all_gt_features['train'].columns)
    print(f"  Feature count: {len(feature_names)}")
    print(f"  Features: {feature_names}")

    # Step 3: Log-transform skewed features
    print("\n[Step 3] Log-transforming skewed features...")
    for source_dict in [all_gt_features, all_unet_features, all_sf_features]:
        for split in source_dict:
            source_dict[split] = log_transform_skewed(source_dict[split])

    # Step 4: Fit StandardScaler on training GT
    print("\n[Step 4] Fitting StandardScaler on training set...")
    scaler = StandardScaler()
    scaler.fit(all_gt_features['train'].values)

    # Apply scaler
    for source_dict, name in [(all_gt_features, 'gt'),
                               (all_unet_features, 'unet'),
                               (all_sf_features, 'segformer')]:
        for split in source_dict:
            df = source_dict[split]
            scaled = scaler.transform(df.values)
            source_dict[split] = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    joblib.dump(scaler, RESULTS_DIR / 'feature_scaler.joblib')

    # Step 5: Save feature CSVs
    print("\n[Step 5] Saving feature CSVs...")
    for source_dict, name in [(all_gt_features, 'gt'),
                               (all_unet_features, 'unet'),
                               (all_sf_features, 'segformer')]:
        combined = pd.concat([source_dict[s] for s in ['train', 'validation', 'test']])
        combined.to_csv(RESULTS_DIR / f'features_{name}.csv')
        print(f"  Saved: features_{name}.csv ({combined.shape})")

    # Save split assignments
    split_assignments = {}
    for split in ['train', 'validation', 'test']:
        for fname in all_gt_features[split].index:
            split_assignments[fname] = split

    with open(RESULTS_DIR / 'feature_split_assignments.json', 'w') as f:
        json.dump(split_assignments, f, indent=2)

    # Step 6: Load severity labels and save combined
    print("\n[Step 6] Preparing severity labels...")
    with open(RESULTS_DIR / 'severity_labels.json', 'r') as f:
        severity_labels = json.load(f)

    gt_combined = pd.concat([all_gt_features[s] for s in ['train', 'validation', 'test']])
    labels_df = pd.DataFrame({
        'filename': gt_combined.index,
        'severity_label': [severity_labels[f]['label'] for f in gt_combined.index],
        'severity_name': [severity_labels[f]['severity'] for f in gt_combined.index],
        'split': [severity_labels[f]['split'] for f in gt_combined.index],
    })
    labels_df.to_csv(RESULTS_DIR / 'severity_labels.csv', index=False)

    # Step 7: Feature-GT correlation analysis
    print("\n[Step 7] Computing GT-prediction feature correlation...")
    gt_train = all_gt_features['train']
    unet_train = all_unet_features['train']
    sf_train = all_sf_features['train']

    correlations = {}
    for feat in feature_names:
        unet_corr = np.corrcoef(gt_train[feat].values, unet_train[feat].values)[0, 1]
        sf_corr = np.corrcoef(gt_train[feat].values, sf_train[feat].values)[0, 1]
        correlations[feat] = {
            'unet_correlation': float(unet_corr) if not np.isnan(unet_corr) else 0.0,
            'segformer_correlation': float(sf_corr) if not np.isnan(sf_corr) else 0.0,
        }

    mean_unet_corr = np.mean([v['unet_correlation'] for v in correlations.values()])
    mean_sf_corr = np.mean([v['segformer_correlation'] for v in correlations.values()])
    print(f"  Mean U-Net feature correlation with GT: {mean_unet_corr:.4f}")
    print(f"  Mean SegFormer feature correlation with GT: {mean_sf_corr:.4f}")

    with open(RESULTS_DIR / 'feature_correlations.json', 'w') as f:
        json.dump(correlations, f, indent=2)

    # Step 8: Visualizations
    print("\n[Step 8] Generating feature distribution plots...")
    plot_feature_distributions(
        all_gt_features['train'], all_unet_features['train'], all_sf_features['train'],
        FIGURES_DIR / 'fig09_feature_distributions.png'
    )

    # Save feature names
    with open(RESULTS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Features per image: {len(feature_names)}")
    print(f"  Total samples: {len(gt_combined)}")
    print(f"  Feature sources: GT, U-Net, SegFormer")


if __name__ == "__main__":
    main()
