"""
07_seg_explainability.py
=========================
SHAP Explainability for Multi-class Severity Classification

Adapted from src/04_explainability.py for multi-class (3 severity levels).
Uses TreeSHAP on the best XGBoost model from experiment results.

Outputs:
- Global importance bar chart (mean |SHAP| across all classes)
- Per-class beeswarm plots (3 plots)
- Dependence plots for top 5 features
- Waterfall plots for 6 cases (2 per severity class)
- Clinical interpretation with feature-to-clinical-name mapping

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "seg_models"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Feature name -> clinical name mapping
FEATURE_CLINICAL_NAMES = {
    'rbc_count': 'RBC Count',
    'rbc_total_area': 'RBC Total Area',
    'rbc_mean_area': 'RBC Mean Size',
    'rbc_max_area': 'RBC Max Size',
    'wbc_count': 'WBC Count',
    'wbc_total_area': 'WBC Total Area',
    'wbc_mean_area': 'WBC Mean Size',
    'wbc_max_area': 'WBC Max Size',
    'bacteria_count': 'Bacteria Count',
    'bacteria_total_area': 'Bacteria Total Area',
    'bacteria_mean_area': 'Bacteria Mean Size',
    'bacteria_max_area': 'Bacteria Max Size',
    'small_epc_count': 'Small Epithelial Count',
    'small_epc_total_area': 'Small Epithelial Area',
    'small_epc_mean_area': 'Small Epithelial Mean Size',
    'small_epc_max_area': 'Small Epithelial Max Size',
    'large_epc_count': 'Large Epithelial Count',
    'large_epc_total_area': 'Large Epithelial Area',
    'large_epc_mean_area': 'Large Epithelial Mean Size',
    'large_epc_max_area': 'Large Epithelial Max Size',
    'epc_sheet_count': 'Epithelial Sheet Count',
    'epc_sheet_total_area': 'Epithelial Sheet Area',
    'epc_sheet_mean_area': 'Epithelial Sheet Mean Size',
    'epc_sheet_max_area': 'Epithelial Sheet Max Size',
    'yeast_count': 'Yeast Count',
    'yeast_total_area': 'Yeast Total Area',
    'yeast_mean_area': 'Yeast Mean Size',
    'yeast_max_area': 'Yeast Max Size',
    'total_cell_count': 'Total Cell Count',
    'cell_density': 'Cell Density',
    'foreground_ratio': 'Foreground Ratio',
    'bacteria_wbc_ratio': 'Bacteria/WBC Ratio',
    'epithelial_total_area': 'Total Epithelial Area',
    'epi_sheet_ratio': 'Epithelial Sheet Ratio',
    'infection_signature': 'Infection Signature',
    'inflammation_score': 'Inflammation Score',
}

SEVERITY_NAMES = ['Mild', 'Moderate', 'Severe']


def load_best_xgboost():
    """Load the best XGBoost model (highest F1-macro from classification results)."""
    with open(RESULTS_DIR / 'classification_results.json', 'r') as f:
        results = json.load(f)

    # Find best XGBoost experiment
    xgb_experiments = {k: v for k, v in results.items() if 'XGB' in k}
    best_exp = max(xgb_experiments, key=lambda k: xgb_experiments[k]['f1_macro'])
    print(f"  Best XGBoost experiment: {best_exp} (F1={xgb_experiments[best_exp]['f1_macro']:.4f})")

    model = joblib.load(MODELS_DIR / f'xgboost_{best_exp}.joblib')
    return model, best_exp


def load_features_and_labels():
    """Load features from the best source (GT for explainability)."""
    features_df = pd.read_csv(RESULTS_DIR / 'features_gt.csv', index_col=0)
    labels_df = pd.read_csv(RESULTS_DIR / 'severity_labels.csv')

    with open(RESULTS_DIR / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)

    with open(RESULTS_DIR / 'feature_split_assignments.json', 'r') as f:
        splits = json.load(f)

    label_map = dict(zip(labels_df['filename'], labels_df['severity_label']))

    train_idx = [f for f in features_df.index if splits.get(f) == 'train']
    test_idx = [f for f in features_df.index if splits.get(f) == 'test']

    X_train = features_df.loc[train_idx].values
    X_test = features_df.loc[test_idx].values
    y_test = np.array([label_map[f] for f in test_idx])

    return X_train, X_test, y_test, feature_names


def plot_global_importance(shap_values_list, feature_names, save_path):
    """Plot mean |SHAP| across all classes."""
    # shap_values_list is a list of 3 arrays (one per class)
    mean_abs_shap = np.zeros(len(feature_names))
    for sv in shap_values_list:
        mean_abs_shap += np.abs(sv).mean(axis=0)
    mean_abs_shap /= len(shap_values_list)

    indices = np.argsort(mean_abs_shap)
    clinical_names = [FEATURE_CLINICAL_NAMES.get(feature_names[i], feature_names[i]) for i in indices]

    # Show top 15
    top_n = min(15, len(indices))
    indices = indices[-top_n:]
    clinical_names = clinical_names[-top_n:]
    values = mean_abs_shap[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdBu_r(values / values.max())
    ax.barh(range(len(indices)), values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(clinical_names)
    ax.set_xlabel('Mean |SHAP Value| (All Severity Classes)')
    ax.set_title('SHAP Feature Importance — Severity Classification')

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_beeswarm(shap_values_list, X, feature_names, save_dir):
    """Plot beeswarm for each severity class."""
    clinical_names = [FEATURE_CLINICAL_NAMES.get(f, f) for f in feature_names]

    for cls_idx, cls_name in enumerate(SEVERITY_NAMES):
        plt.figure(figsize=(12, 8))
        explanation = shap.Explanation(
            values=shap_values_list[cls_idx],
            base_values=np.zeros(len(shap_values_list[cls_idx])),
            data=X,
            feature_names=clinical_names,
        )
        shap.plots.beeswarm(explanation, max_display=15, show=False)
        plt.title(f'SHAP Values for {cls_name} Severity', fontsize=14)
        plt.tight_layout()
        save_path = save_dir / f'fig_shap_beeswarm_{cls_name.lower()}.png'
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def plot_dependence(shap_values_list, X, feature_names, save_dir, top_n=5):
    """Plot dependence plots for top features (using class 2 = Severe)."""
    mean_abs = np.abs(shap_values_list[2]).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-top_n:][::-1]

    for idx in top_indices:
        feat = feature_names[idx]
        clinical = FEATURE_CLINICAL_NAMES.get(feat, feat)

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(X[:, idx], shap_values_list[2][:, idx],
                             c=X[:, idx], cmap='RdBu_r', alpha=0.6,
                             edgecolors='black', linewidth=0.3, s=30)
        plt.colorbar(scatter, label=clinical)
        ax.set_xlabel(clinical)
        ax.set_ylabel(f'SHAP Value (Severe Class)')
        ax.set_title(f'Dependence: {clinical}')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = save_dir / f'fig_shap_dep_{feat}.png'
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def plot_waterfall_cases(explainer, shap_values_list, X, y, feature_names, save_dir):
    """Plot waterfall plots for 6 cases (2 per severity class)."""
    clinical_names = [FEATURE_CLINICAL_NAMES.get(f, f) for f in feature_names]

    for cls_idx, cls_name in enumerate(SEVERITY_NAMES):
        cls_mask = (y == cls_idx)
        cls_indices = np.where(cls_mask)[0]

        if len(cls_indices) < 2:
            continue

        for case_i, sample_idx in enumerate(cls_indices[:2]):
            fig, ax = plt.subplots(figsize=(10, 6))

            explanation = shap.Explanation(
                values=shap_values_list[cls_idx][sample_idx],
                base_values=explainer.expected_value[cls_idx],
                data=X[sample_idx],
                feature_names=clinical_names,
            )

            shap.plots.waterfall(explanation, max_display=12, show=False)
            plt.title(f'{cls_name} Case {case_i + 1} (Sample {sample_idx})', fontsize=12)
            plt.tight_layout()
            save_path = save_dir / f'fig_shap_waterfall_{cls_name.lower()}_case{case_i + 1}.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved: {save_path}")


def generate_importance_table(shap_values_list, feature_names) -> pd.DataFrame:
    """Generate SHAP importance ranking table."""
    mean_abs_shap = np.zeros(len(feature_names))
    for sv in shap_values_list:
        mean_abs_shap += np.abs(sv).mean(axis=0)
    mean_abs_shap /= len(shap_values_list)

    rows = []
    indices = np.argsort(mean_abs_shap)[::-1]
    for rank, idx in enumerate(indices, 1):
        feat = feature_names[idx]
        rows.append({
            'Rank': rank,
            'Feature': feat,
            'Clinical Name': FEATURE_CLINICAL_NAMES.get(feat, feat),
            'Mean |SHAP|': f'{mean_abs_shap[idx]:.4f}',
        })

    df = pd.DataFrame(rows)
    return df


def main():
    print("=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Step 1: Load model and data
    print("\n[Step 1] Loading model and data...")
    model, best_exp = load_best_xgboost()
    X_train, X_test, y_test, feature_names = load_features_and_labels()

    # Step 2: Compute SHAP values
    print("\n[Step 2] Computing SHAP values (TreeSHAP)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values is a list of 3 arrays for multi-class
    if isinstance(shap_values, list):
        shap_values_list = shap_values
    else:
        # Single array -> split by class
        shap_values_list = [shap_values[:, :, i] for i in range(3)]

    print(f"  SHAP values: {len(shap_values_list)} classes, shape per class: {shap_values_list[0].shape}")

    # Step 3: Global importance
    print("\n[Step 3] Generating global importance plot...")
    plot_global_importance(shap_values_list, feature_names,
                          FIGURES_DIR / 'fig12_shap_global_importance.png')

    # Step 4: Per-class beeswarm
    print("\n[Step 4] Generating per-class beeswarm plots...")
    plot_per_class_beeswarm(shap_values_list, X_test, feature_names, FIGURES_DIR)

    # Step 5: Dependence plots
    print("\n[Step 5] Generating dependence plots (top 5 features)...")
    plot_dependence(shap_values_list, X_test, feature_names, FIGURES_DIR, top_n=5)

    # Step 6: Waterfall cases
    print("\n[Step 6] Generating waterfall plots (6 cases)...")
    plot_waterfall_cases(explainer, shap_values_list, X_test, y_test, feature_names, FIGURES_DIR)

    # Step 7: Save importance table
    print("\n[Step 7] Generating importance table...")
    importance_df = generate_importance_table(shap_values_list, feature_names)
    importance_df.to_csv(RESULTS_DIR / 'shap_importance_ranking.csv', index=False)
    print(importance_df.head(10).to_string(index=False))

    # Save SHAP values
    np.savez(RESULTS_DIR / 'shap_values_multiclass.npz',
             **{f'class_{i}': sv for i, sv in enumerate(shap_values_list)},
             expected_values=np.array(explainer.expected_value))

    # Save results JSON
    results = {
        'model': best_exp,
        'expected_values': [float(v) for v in explainer.expected_value],
        'top_10_features': importance_df.head(10).to_dict(orient='records'),
    }
    with open(RESULTS_DIR / 'shap_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nTop 5 Most Important Features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['Rank']}. {row['Clinical Name']}: {row['Mean |SHAP|']}")


if __name__ == "__main__":
    main()
