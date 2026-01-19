"""
02_xgboost_baseline.py
======================
UTI Prediction - XGBoost Baseline Model

This script performs:
1. XGBoost model training with conservative hyperparameters for small dataset
2. Hyperparameter optimization using Optuna
3. 5-fold stratified cross-validation
4. Bootstrap confidence intervals for all metrics
5. Model evaluation and comparison
6. Feature importance analysis
7. Learning curve generation

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import warnings
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, matthews_corrcoef,
    average_precision_score
)
from sklearn.utils import resample

import xgboost as xgb
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

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], dict]:
    """
    Load preprocessed train/test data.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names, class_weights
    """
    # Load train/test split
    data = np.load(RESULTS_DIR / 'train_test_split.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # Load feature names
    with open(RESULTS_DIR / 'feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')

    # Load class weights
    class_weights = joblib.load(RESULTS_DIR / 'class_weights.joblib')

    print(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Features: {feature_names}")
    print(f"Class weights: {class_weights}")

    return X_train, X_test, y_train, y_test, feature_names, class_weights


def get_conservative_xgb_params() -> dict:
    """
    Get conservative XGBoost parameters for small dataset (n=120).

    Returns
    -------
    dict
        XGBoost parameters optimized for small datasets
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,  # Shallow trees to prevent overfitting
        'min_child_weight': 5,  # Higher value for more conservative
        'n_estimators': 100,
        'learning_rate': 0.05,  # Lower learning rate
        'subsample': 0.8,  # Slight subsampling
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,  # L1 regularization
        'reg_lambda': 2.0,  # L2 regularization
        'gamma': 0.1,  # Minimum loss reduction for split
        'random_state': RANDOM_STATE,
        'use_label_encoder': False,
        'verbosity': 0
    }
    return params


def hyperparameter_search(X_train: np.ndarray, y_train: np.ndarray,
                          class_weights: dict, n_trials: int = 50) -> dict:
    """
    Perform hyperparameter optimization using Optuna.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    class_weights : dict
        Class weights for imbalanced data
    n_trials : int
        Number of optimization trials

    Returns
    -------
    dict
        Best hyperparameters
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'random_state': RANDOM_STATE,
                'use_label_encoder': False,
                'verbosity': 0
            }

            # Calculate scale_pos_weight for XGBoost
            scale_pos_weight = class_weights[0] / class_weights[1]
            params['scale_pos_weight'] = scale_pos_weight

            model = xgb.XGBClassifier(**params)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest trial AUC: {study.best_trial.value:.4f}")
        print(f"Best parameters: {study.best_trial.params}")

        return study.best_trial.params

    except ImportError:
        print("Optuna not available, using default parameters")
        return get_conservative_xgb_params()


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray,
                        class_weights: dict, params: dict = None) -> xgb.XGBClassifier:
    """
    Train XGBoost model with given parameters.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    class_weights : dict
        Class weights
    params : dict, optional
        Model parameters

    Returns
    -------
    xgb.XGBClassifier
        Trained model
    """
    if params is None:
        params = get_conservative_xgb_params()

    # Add scale_pos_weight for class imbalance
    scale_pos_weight = class_weights[0] / class_weights[1]

    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': params.get('max_depth', 3),
        'min_child_weight': params.get('min_child_weight', 5),
        'n_estimators': params.get('n_estimators', 100),
        'learning_rate': params.get('learning_rate', 0.05),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'reg_alpha': params.get('reg_alpha', 1.0),
        'reg_lambda': params.get('reg_lambda', 2.0),
        'gamma': params.get('gamma', 0.1),
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_STATE,
        'use_label_encoder': False,
        'verbosity': 0
    }

    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    return model


def bootstrap_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: np.ndarray, n_bootstrap: int = 1000,
                                   ci: float = 0.95) -> dict:
    """
    Calculate bootstrap confidence intervals for all metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level

    Returns
    -------
    dict
        Dictionary with metric values and confidence intervals
    """
    metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'auc': [], 'mcc': [], 'ap': []
    }

    for _ in range(n_bootstrap):
        indices = resample(range(len(y_true)), random_state=None)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['auc'].append(roc_auc_score(y_true_boot, y_prob_boot))
        metrics['mcc'].append(matthews_corrcoef(y_true_boot, y_pred_boot))
        metrics['ap'].append(average_precision_score(y_true_boot, y_prob_boot))

    alpha = (1 - ci) / 2
    results = {}

    for metric_name, values in metrics.items():
        values = np.array(values)
        results[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, alpha * 100),
            'ci_upper': np.percentile(values, (1 - alpha) * 100)
        }

    return results


def cross_validate_model(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray,
                         n_splits: int = 5) -> dict:
    """
    Perform stratified k-fold cross-validation.

    Parameters
    ----------
    model : xgb.XGBClassifier
        XGBoost model (will be cloned for each fold)
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_splits : int
        Number of CV folds

    Returns
    -------
    dict
        Cross-validation results
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'auc': [], 'mcc': []
    }

    fold_predictions = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Clone and train model
        model_clone = xgb.XGBClassifier(**model.get_params())
        model_clone.fit(X_train_fold, y_train_fold)

        # Predictions
        y_pred = model_clone.predict(X_val_fold)
        y_prob = model_clone.predict_proba(X_val_fold)[:, 1]

        # Calculate metrics
        cv_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        cv_results['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
        cv_results['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
        cv_results['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
        cv_results['auc'].append(roc_auc_score(y_val_fold, y_prob))
        cv_results['mcc'].append(matthews_corrcoef(y_val_fold, y_pred))

        fold_predictions.append({
            'fold': fold + 1,
            'y_true': y_val_fold,
            'y_pred': y_pred,
            'y_prob': y_prob
        })

    # Calculate summary statistics
    summary = {}
    for metric, values in cv_results.items():
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values,
            'ci_95': (np.mean(values) - 1.96 * np.std(values) / np.sqrt(n_splits),
                      np.mean(values) + 1.96 * np.std(values) / np.sqrt(n_splits))
        }

    return summary, fold_predictions


def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """
    Comprehensive model evaluation on test set.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels

    Returns
    -------
    dict
        Evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'ap': average_precision_score(y_test, y_prob)
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

    # Specificity
    metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])

    # Bootstrap CI
    bootstrap_results = bootstrap_confidence_intervals(y_test, y_pred, y_prob)
    metrics['bootstrap_ci'] = bootstrap_results

    return metrics, y_pred, y_prob


def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray,
                   auc_score: float, save_path: Path) -> None:
    """
    Plot ROC curve with AUC score.

    Parameters
    ----------
    y_test : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    auc_score : float
        AUC score
    save_path : Path
        Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='#3498db', lw=2,
            label=f'XGBoost (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - XGBoost', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add AUC annotation
    ax.text(0.6, 0.2, f'AUC = {auc_score:.4f}',
            fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_recall_curve(y_test: np.ndarray, y_prob: np.ndarray,
                                ap_score: float, save_path: Path) -> None:
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_test : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    ap_score : float
        Average precision score
    save_path : Path
        Path to save figure
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(recall, precision, color='#e74c3c', lw=2,
            label=f'XGBoost (AP = {ap_score:.4f})')

    # Baseline (random classifier)
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline = {baseline:.2f}')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - XGBoost', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    save_path : Path
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=ax, annot_kws={'size': 16})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - XGBoost', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(model: xgb.XGBClassifier, feature_names: list,
                            save_path: Path) -> None:
    """
    Plot feature importance from XGBoost model.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained model
    feature_names : list
        Feature names
    save_path : Path
        Path to save figure
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlBu_r(importance[indices] / importance.max())
    ax.barh(range(len(indices)), importance[indices], color=colors, edgecolor='black')

    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('XGBoost Feature Importance', fontsize=14)

    # Add value labels
    for i, v in enumerate(importance[indices]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curve(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray,
                        save_path: Path) -> None:
    """
    Plot learning curve to assess overfitting.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Model to evaluate
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    save_path : Path
        Path to save figure
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 8),
        scoring='roc_auc'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='#3498db')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color='#e74c3c')

    ax.plot(train_sizes, train_mean, 'o-', color='#3498db', lw=2, label='Training Score')
    ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', lw=2, label='Validation Score')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('Learning Curve - XGBoost', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation for final gap
    gap = train_mean[-1] - val_mean[-1]
    ax.annotate(f'Gap: {gap:.3f}', xy=(train_sizes[-1], (train_mean[-1] + val_mean[-1])/2),
                fontsize=10, ha='left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cv_results(cv_summary: dict, save_path: Path) -> None:
    """
    Plot cross-validation results with error bars.

    Parameters
    ----------
    cv_summary : dict
        Cross-validation summary statistics
    save_path : Path
        Path to save figure
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
    means = [cv_summary[m]['mean'] for m in metrics]
    stds = [cv_summary[m]['std'] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics)))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('5-Fold Cross-Validation Results - XGBoost', fontsize=14)
    ax.set_ylim(0, 1.1)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_metrics_table(test_metrics: dict, cv_summary: dict) -> pd.DataFrame:
    """
    Generate comprehensive metrics table for manuscript.

    Parameters
    ----------
    test_metrics : dict
        Test set evaluation metrics
    cv_summary : dict
        Cross-validation summary

    Returns
    -------
    pd.DataFrame
        Metrics table
    """
    rows = []

    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']

    for metric in metrics_list:
        row = {
            'Metric': metric.upper() if metric != 'auc' else 'AUC-ROC',
            'Test Set': f"{test_metrics[metric]:.4f}",
            'CV Mean': f"{cv_summary[metric]['mean']:.4f}",
            'CV Std': f"{cv_summary[metric]['std']:.4f}",
            '95% CI': f"[{test_metrics['bootstrap_ci'][metric]['ci_lower']:.4f}, {test_metrics['bootstrap_ci'][metric]['ci_upper']:.4f}]"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'xgboost_performance_metrics.csv', index=False)
    print(f"Saved: {TABLES_DIR / 'xgboost_performance_metrics.csv'}")

    return df


def generate_hyperparameter_table(params: dict) -> pd.DataFrame:
    """
    Generate hyperparameter table for manuscript.

    Parameters
    ----------
    params : dict
        Model hyperparameters

    Returns
    -------
    pd.DataFrame
        Hyperparameter table
    """
    rows = []

    param_descriptions = {
        'max_depth': 'Maximum tree depth',
        'min_child_weight': 'Minimum child weight',
        'n_estimators': 'Number of boosting rounds',
        'learning_rate': 'Learning rate (eta)',
        'subsample': 'Subsample ratio',
        'colsample_bytree': 'Column subsample ratio',
        'reg_alpha': 'L1 regularization',
        'reg_lambda': 'L2 regularization',
        'gamma': 'Minimum loss reduction',
        'scale_pos_weight': 'Class weight ratio'
    }

    for param, value in params.items():
        if param in param_descriptions:
            rows.append({
                'Parameter': param,
                'Value': f"{value:.4f}" if isinstance(value, float) else str(value),
                'Description': param_descriptions[param]
            })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'xgboost_hyperparameters.csv', index=False)
    print(f"Saved: {TABLES_DIR / 'xgboost_hyperparameters.csv'}")

    return df


def main():
    """Main XGBoost training pipeline."""
    print("=" * 60)
    print("UTI PREDICTION - XGBOOST BASELINE MODEL")
    print("=" * 60)

    # Track training time
    start_time = time.time()

    # Step 1: Load data
    print("\n[Step 1] Loading preprocessed data...")
    X_train, X_test, y_train, y_test, feature_names, class_weights = load_processed_data()

    # Step 2: Hyperparameter optimization
    print("\n[Step 2] Hyperparameter optimization...")
    try:
        import optuna
        best_params = hyperparameter_search(X_train, y_train, class_weights, n_trials=50)
    except ImportError:
        print("Optuna not available, using default conservative parameters")
        best_params = get_conservative_xgb_params()

    # Step 3: Train final model
    print("\n[Step 3] Training final model...")
    model = train_xgboost_model(X_train, y_train, class_weights, best_params)

    # Step 4: Cross-validation
    print("\n[Step 4] Performing 5-fold cross-validation...")
    cv_summary, fold_predictions = cross_validate_model(model, X_train, y_train)

    print("\nCross-Validation Results:")
    for metric, stats in cv_summary.items():
        print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Step 5: Evaluate on test set
    print("\n[Step 5] Evaluating on test set...")
    test_metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    print("\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")

    # Step 6: Generate visualizations
    print("\n[Step 6] Generating visualizations...")
    plot_roc_curve(y_test, y_prob, test_metrics['auc'],
                   FIGURES_DIR / 'fig06_xgboost_roc_curve.png')
    plot_precision_recall_curve(y_test, y_prob, test_metrics['ap'],
                                FIGURES_DIR / 'fig07_xgboost_pr_curve.png')
    plot_confusion_matrix(np.array(test_metrics['confusion_matrix']),
                          FIGURES_DIR / 'fig08_xgboost_confusion_matrix.png')
    plot_feature_importance(model, feature_names,
                            FIGURES_DIR / 'fig09_xgboost_feature_importance.png')
    plot_learning_curve(model, X_train, y_train,
                        FIGURES_DIR / 'fig10_xgboost_learning_curve.png')
    plot_cv_results(cv_summary, FIGURES_DIR / 'fig11_xgboost_cv_results.png')

    # Step 7: Generate tables
    print("\n[Step 7] Generating tables...")
    metrics_table = generate_metrics_table(test_metrics, cv_summary)
    hyperparams_table = generate_hyperparameter_table(model.get_params())

    # Step 8: Save model and results
    print("\n[Step 8] Saving model and results...")

    # Save model
    model.save_model(str(MODELS_DIR / 'xgboost_model.json'))
    joblib.dump(model, MODELS_DIR / 'xgboost_model.joblib')
    print(f"Saved: {MODELS_DIR / 'xgboost_model.json'}")
    print(f"Saved: {MODELS_DIR / 'xgboost_model.joblib'}")

    # Save complete results
    training_time = time.time() - start_time

    results = {
        'model': 'XGBoost',
        'training_time_seconds': training_time,
        'test_metrics': test_metrics,
        'cv_summary': {k: {'mean': v['mean'], 'std': v['std']}
                       for k, v in cv_summary.items()},
        'best_params': best_params,
        'feature_names': feature_names,
        'feature_importance': dict(zip(feature_names, model.feature_importances_.tolist()))
    }

    with open(RESULTS_DIR / 'xgboost_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'xgboost_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"\nBest Performance (Test Set):")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\nTop 3 Important Features:")
    importance_sorted = sorted(zip(feature_names, model.feature_importances_),
                               key=lambda x: x[1], reverse=True)
    for feat, imp in importance_sorted[:3]:
        print(f"  {feat}: {imp:.4f}")

    return model, results


if __name__ == "__main__":
    model, results = main()
