"""
04_explainability.py
====================
UTI Prediction - Model Explainability with SHAP

This script performs:
1. SHAP TreeExplainer for XGBoost
2. Global feature importance analysis
3. Local explanations (individual predictions)
4. Feature dependence plots
5. Clinical interpretation generation

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import joblib

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

# Plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Feature name mapping for clinical interpretation
FEATURE_CLINICAL_NAMES = {
    'temperature': 'Body Temperature (°C)',
    'nausea': 'Nausea',
    'lumbar_pain': 'Lumbar Pain',
    'urine_pushing': 'Urinary Urgency',
    'micturition_pains': 'Painful Urination',
    'burning_urethra': 'Burning/Itching Urethra',
    'fever': 'Fever (>37.5°C)',
    'high_fever': 'High Fever (>38.5°C)',
    'symptom_count': 'Total Symptom Count'
}


def load_data_and_model() -> Tuple:
    """
    Load preprocessed data and trained XGBoost model.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names, model
    """
    # Load data
    data = np.load(RESULTS_DIR / 'train_test_split.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # Load feature names
    with open(RESULTS_DIR / 'feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')

    # Load model
    model = joblib.load(MODELS_DIR / 'xgboost_model.joblib')

    # Load processed data for full context
    df = pd.read_csv(RESULTS_DIR / 'processed_data.csv')

    print(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Features: {feature_names}")

    return X_train, X_test, y_train, y_test, feature_names, model, df


def compute_shap_values(model, X: np.ndarray, feature_names: List[str]) -> Tuple:
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names

    Returns
    -------
    tuple
        explainer, shap_values
    """
    print("Computing SHAP values...")

    # Use TreeExplainer for XGBoost (most efficient)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(f"SHAP values shape: {shap_values.shape}")

    return explainer, shap_values


def analyze_global_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Analyze global feature importance from SHAP values.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix
    feature_names : list
        Feature names

    Returns
    -------
    pd.DataFrame
        Feature importance dataframe
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'clinical_name': [FEATURE_CLINICAL_NAMES.get(f, f) for f in feature_names],
        'mean_abs_shap': mean_abs_shap,
        'std_shap': np.abs(shap_values).std(axis=0),
        'mean_shap': shap_values.mean(axis=0)  # Direction of effect
    })

    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)

    return importance_df


def plot_shap_summary_bar(shap_values: np.ndarray, feature_names: List[str],
                          save_path: Path) -> None:
    """
    Plot SHAP summary bar chart (global importance).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    feature_names : list
        Feature names
    save_path : Path
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs_shap)

    clinical_names = [FEATURE_CLINICAL_NAMES.get(feature_names[i], feature_names[i])
                      for i in indices]

    colors = plt.cm.RdBu_r(mean_abs_shap[indices] / mean_abs_shap.max())

    bars = ax.barh(range(len(indices)), mean_abs_shap[indices],
                   color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(clinical_names)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('SHAP Feature Importance - XGBoost Model', fontsize=14)

    # Add value labels
    for i, v in enumerate(mean_abs_shap[indices]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_beeswarm(shap_values: np.ndarray, X: np.ndarray,
                       feature_names: List[str], save_path: Path) -> None:
    """
    Plot SHAP beeswarm plot showing feature value effects.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    save_path : Path
        Path to save figure
    """
    # Create DataFrame for plotting
    X_df = pd.DataFrame(X, columns=feature_names)

    # Rename to clinical names
    X_df_clinical = X_df.rename(columns=FEATURE_CLINICAL_NAMES)
    feature_names_clinical = [FEATURE_CLINICAL_NAMES.get(f, f) for f in feature_names]

    plt.figure(figsize=(12, 8))

    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),  # Placeholder
        data=X_df_clinical.values,
        feature_names=feature_names_clinical
    )

    shap.plots.beeswarm(shap_explanation, max_display=len(feature_names), show=False)

    plt.title('SHAP Beeswarm Plot - Feature Impact on UTI Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_dependence(shap_values: np.ndarray, X: np.ndarray,
                         feature_names: List[str], top_n: int = 3,
                         save_dir: Path = FIGURES_DIR) -> None:
    """
    Plot SHAP dependence plots for top features.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    top_n : int
        Number of top features to plot
    save_dir : Path
        Directory to save figures
    """
    # Get top features by importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    for idx in top_indices:
        feature = feature_names[idx]
        clinical_name = FEATURE_CLINICAL_NAMES.get(feature, feature)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Find best interaction feature
        interaction_idx = None
        if len(feature_names) > 1:
            correlations = []
            for j in range(len(feature_names)):
                if j != idx:
                    corr = np.corrcoef(X[:, idx], shap_values[:, idx])[0, 1]
                    correlations.append((j, abs(corr) if not np.isnan(corr) else 0))
            if correlations:
                interaction_idx = max(correlations, key=lambda x: x[1])[0]

        scatter = ax.scatter(X[:, idx], shap_values[:, idx],
                            c=X[:, interaction_idx] if interaction_idx else 'steelblue',
                            cmap='RdBu_r', alpha=0.6, edgecolors='black', linewidth=0.3)

        if interaction_idx:
            interaction_name = FEATURE_CLINICAL_NAMES.get(feature_names[interaction_idx],
                                                          feature_names[interaction_idx])
            plt.colorbar(scatter, label=interaction_name)

        ax.set_xlabel(clinical_name, fontsize=12)
        ax.set_ylabel(f'SHAP Value for {clinical_name}', fontsize=12)
        ax.set_title(f'SHAP Dependence Plot: {clinical_name}', fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = save_dir / f'fig_shap_dependence_{feature}.png'
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def plot_shap_waterfall(explainer, shap_values: np.ndarray, X: np.ndarray,
                        y: np.ndarray, feature_names: List[str],
                        n_cases: int = 6, save_dir: Path = FIGURES_DIR) -> None:
    """
    Plot SHAP waterfall plots for individual cases.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer
    shap_values : np.ndarray
        SHAP values
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True labels
    feature_names : list
        Feature names
    n_cases : int
        Number of cases to plot
    save_dir : Path
        Directory to save figures
    """
    clinical_names = [FEATURE_CLINICAL_NAMES.get(f, f) for f in feature_names]

    # Select diverse cases: true positives, true negatives, etc.
    predictions = (shap_values.sum(axis=1) + explainer.expected_value) > 0

    # Try to get balanced cases
    tp_indices = np.where((y == 1) & (predictions == 1))[0]
    tn_indices = np.where((y == 0) & (predictions == 0))[0]
    fp_indices = np.where((y == 0) & (predictions == 1))[0]
    fn_indices = np.where((y == 1) & (predictions == 0))[0]

    case_indices = []
    for indices, label in [(tp_indices, 'TP'), (tn_indices, 'TN'),
                           (fp_indices, 'FP'), (fn_indices, 'FN')]:
        if len(indices) > 0:
            case_indices.append((indices[0], label))
        if len(case_indices) >= n_cases:
            break

    # Fill remaining with random cases
    while len(case_indices) < min(n_cases, len(X)):
        idx = np.random.randint(0, len(X))
        if idx not in [c[0] for c in case_indices]:
            label = 'TP' if y[idx] == 1 and predictions[idx] == 1 else \
                    'TN' if y[idx] == 0 and predictions[idx] == 0 else \
                    'FP' if y[idx] == 0 and predictions[idx] == 1 else 'FN'
            case_indices.append((idx, label))

    for i, (idx, case_type) in enumerate(case_indices):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create explanation object for this sample
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X[idx],
            feature_names=clinical_names
        )

        shap.plots.waterfall(explanation, max_display=len(feature_names), show=False)

        true_label = "UTI Positive" if y[idx] == 1 else "UTI Negative"
        pred_label = "Positive" if predictions[idx] else "Negative"
        plt.title(f'Case {i+1} ({case_type}): True={true_label}, Pred={pred_label}',
                  fontsize=12)

        plt.tight_layout()
        save_path = save_dir / f'fig_shap_waterfall_case{i+1}_{case_type}.png'
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def plot_shap_force(explainer, shap_values: np.ndarray, X: np.ndarray,
                    feature_names: List[str], save_path: Path) -> None:
    """
    Plot SHAP force plot for multiple instances.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer
    shap_values : np.ndarray
        SHAP values
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    save_path : Path
        Path to save figure
    """
    clinical_names = [FEATURE_CLINICAL_NAMES.get(f, f) for f in feature_names]

    # Create force plot for first 20 samples (or all if less)
    n_samples = min(20, len(X))

    shap.initjs()

    # Generate and save HTML
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[:n_samples],
        pd.DataFrame(X[:n_samples], columns=clinical_names),
        show=False
    )

    # Save as HTML
    html_path = save_path.with_suffix('.html')
    shap.save_html(str(html_path), force_plot)
    print(f"Saved: {html_path}")


def generate_clinical_interpretation(importance_df: pd.DataFrame,
                                    shap_values: np.ndarray,
                                    feature_names: List[str]) -> str:
    """
    Generate clinical interpretation of SHAP results.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    shap_values : np.ndarray
        SHAP values
    feature_names : list
        Feature names

    Returns
    -------
    str
        Clinical interpretation text
    """
    interpretation = []

    interpretation.append("# Clinical Interpretation of UTI Prediction Model\n")
    interpretation.append("## Overview\n")
    interpretation.append("The SHAP (SHapley Additive exPlanations) analysis provides "
                         "insight into how the XGBoost model makes UTI predictions based "
                         "on clinical symptoms and signs.\n")

    interpretation.append("\n## Feature Importance Rankings\n")

    for i, row in importance_df.head(5).iterrows():
        clinical_name = row['clinical_name']
        mean_shap = row['mean_shap']
        mean_abs = row['mean_abs_shap']
        direction = "increases" if mean_shap > 0 else "decreases"

        interpretation.append(f"### {row['rank']}. {clinical_name}\n")
        interpretation.append(f"- **Mean |SHAP|**: {mean_abs:.4f}\n")
        interpretation.append(f"- **Effect Direction**: Higher values {direction} UTI risk\n")

        # Feature-specific clinical notes
        if row['feature'] == 'symptom_count':
            interpretation.append("- **Clinical Significance**: The total number of symptoms "
                                "present is the strongest predictor, reflecting the cumulative "
                                "evidence for UTI diagnosis.\n")
        elif row['feature'] == 'urine_pushing':
            interpretation.append("- **Clinical Significance**: Urinary urgency is a classic "
                                "symptom of lower urinary tract infection (cystitis).\n")
        elif row['feature'] == 'micturition_pains':
            interpretation.append("- **Clinical Significance**: Dysuria (painful urination) "
                                "is a hallmark symptom of UTI.\n")
        elif row['feature'] == 'temperature':
            interpretation.append("- **Clinical Significance**: Elevated temperature suggests "
                                "systemic infection, potentially indicating upper UTI.\n")
        elif row['feature'] == 'lumbar_pain':
            interpretation.append("- **Clinical Significance**: Lumbar/flank pain may indicate "
                                "pyelonephritis (kidney infection) rather than simple cystitis.\n")

        interpretation.append("\n")

    interpretation.append("\n## Clinical Recommendations\n")
    interpretation.append("1. **Symptom Assessment**: A comprehensive symptom count is crucial "
                         "for accurate UTI prediction.\n")
    interpretation.append("2. **Urinary Symptoms**: Urinary urgency and painful urination "
                         "should be primary screening questions.\n")
    interpretation.append("3. **Fever Monitoring**: Temperature elevation helps distinguish "
                         "between uncomplicated and complicated UTI.\n")
    interpretation.append("4. **Pain Location**: Lumbar pain suggests upper UTI requiring "
                         "more aggressive treatment.\n")

    interpretation.append("\n## Model Limitations\n")
    interpretation.append("- Small training dataset (n=120) - results may not generalize\n")
    interpretation.append("- Binary features may miss symptom severity gradations\n")
    interpretation.append("- No demographic features (age, sex, history) included\n")
    interpretation.append("- External validation required before clinical deployment\n")

    return "\n".join(interpretation)


def generate_shap_summary_table(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate SHAP summary table for manuscript.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance dataframe

    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    summary = importance_df[['rank', 'clinical_name', 'mean_abs_shap', 'mean_shap', 'std_shap']].copy()
    summary.columns = ['Rank', 'Feature', 'Mean |SHAP|', 'Mean SHAP', 'Std SHAP']

    # Add effect direction
    summary['Effect'] = summary['Mean SHAP'].apply(
        lambda x: 'Positive (+UTI risk)' if x > 0 else 'Negative (-UTI risk)'
    )

    summary.to_csv(TABLES_DIR / 'shap_feature_importance.csv', index=False)
    print(f"Saved: {TABLES_DIR / 'shap_feature_importance.csv'}")

    return summary


def main():
    """Main explainability analysis pipeline."""
    print("=" * 60)
    print("UTI PREDICTION - SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Step 1: Load data and model
    print("\n[Step 1] Loading data and model...")
    X_train, X_test, y_train, y_test, feature_names, model, df = load_data_and_model()

    # Step 2: Compute SHAP values
    print("\n[Step 2] Computing SHAP values...")
    explainer, shap_values_train = compute_shap_values(model, X_train, feature_names)
    _, shap_values_test = compute_shap_values(model, X_test, feature_names)

    # Step 3: Analyze global importance
    print("\n[Step 3] Analyzing global feature importance...")
    importance_df = analyze_global_importance(shap_values_train, feature_names)
    print("\nTop 5 Features by SHAP Importance:")
    print(importance_df[['rank', 'clinical_name', 'mean_abs_shap']].head().to_string(index=False))

    # Step 4: Generate visualizations
    print("\n[Step 4] Generating SHAP visualizations...")

    # Summary bar plot
    plot_shap_summary_bar(shap_values_train, feature_names,
                         FIGURES_DIR / 'fig16_shap_importance_bar.png')

    # Beeswarm plot
    plot_shap_beeswarm(shap_values_train, X_train, feature_names,
                      FIGURES_DIR / 'fig17_shap_beeswarm.png')

    # Dependence plots for top 3 features
    print("\n[Step 5] Generating dependence plots...")
    plot_shap_dependence(shap_values_train, X_train, feature_names, top_n=3)

    # Waterfall plots for individual cases
    print("\n[Step 6] Generating waterfall plots for individual cases...")
    plot_shap_waterfall(explainer, shap_values_test, X_test, y_test,
                       feature_names, n_cases=4)

    # Step 7: Generate tables and interpretation
    print("\n[Step 7] Generating summary table and clinical interpretation...")

    summary_table = generate_shap_summary_table(importance_df)
    print("\nSHAP Feature Importance Summary:")
    print(summary_table.to_string(index=False))

    # Clinical interpretation
    interpretation = generate_clinical_interpretation(importance_df, shap_values_train,
                                                     feature_names)

    interpretation_path = RESULTS_DIR / 'clinical_interpretation.md'
    with open(interpretation_path, 'w') as f:
        f.write(interpretation)
    print(f"Saved: {interpretation_path}")

    # Save SHAP values for later use
    np.savez(RESULTS_DIR / 'shap_values.npz',
             shap_values_train=shap_values_train,
             shap_values_test=shap_values_test,
             expected_value=explainer.expected_value)
    print(f"Saved: {RESULTS_DIR / 'shap_values.npz'}")

    # Save complete results
    results = {
        'expected_value': float(explainer.expected_value),
        'feature_importance': importance_df.to_dict(orient='records'),
        'top_features': importance_df.head(5)['feature'].tolist()
    }

    with open(RESULTS_DIR / 'shap_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'shap_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nKey Findings:")
    print(f"  - Expected value (base rate): {explainer.expected_value:.4f}")
    print(f"\n  Top 3 Most Important Features:")
    for i, row in importance_df.head(3).iterrows():
        direction = "+" if row['mean_shap'] > 0 else "-"
        print(f"    {row['rank']}. {row['clinical_name']}: {row['mean_abs_shap']:.4f} ({direction})")

    print(f"\nOutputs saved to:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Tables: {TABLES_DIR}")
    print(f"  - Clinical interpretation: {interpretation_path}")

    return importance_df, shap_values_train, explainer


if __name__ == "__main__":
    importance_df, shap_values, explainer = main()
