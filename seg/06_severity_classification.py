"""
06_severity_classification.py
==============================
XGBoost + FT-Transformer Severity Classification (6 Experiments)

E1: GT masks -> XGBoost       (Oracle upper bound)
E2: GT masks -> FT-Transformer (Oracle upper bound)
E3: U-Net masks -> XGBoost    (Realistic pipeline)
E4: U-Net masks -> FT-Transformer
E5: SegFormer masks -> XGBoost
E6: SegFormer masks -> FT-Transformer

Reuses patterns from src/02_xgboost_baseline.py and src/03_ft_transformer.py

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from sklearn.utils import resample

import xgboost as xgb
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seg.models.losses import MultiFocalLoss

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "seg_models"


# ============================================================================
# Inline FT-Transformer (adapted from src/03_ft_transformer.py for multi-class)
# ============================================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_o(out), attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_w = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


class FTTransformerMultiClass(nn.Module):
    """FT-Transformer adapted for 3-class severity classification."""

    def __init__(self, n_features, d_token=48, n_layers=3, n_heads=4,
                 d_ff=96, dropout=0.3, n_classes=3):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.Dropout(dropout), nn.Linear(d_token, n_classes)
        )

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            x, _ = layer(x)
        return self.head(x[:, 0, :])


# ============================================================================
# Data Loading
# ============================================================================
def load_experiment_data(feature_source: str) -> Tuple:
    """
    Load features and labels for a given feature source.

    Returns X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    features_df = pd.read_csv(RESULTS_DIR / f'features_{feature_source}.csv', index_col=0)
    labels_df = pd.read_csv(RESULTS_DIR / 'severity_labels.csv')

    with open(RESULTS_DIR / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)

    # Split by assignment
    with open(RESULTS_DIR / 'feature_split_assignments.json', 'r') as f:
        splits = json.load(f)

    label_map = dict(zip(labels_df['filename'], labels_df['severity_label']))

    train_idx = [f for f in features_df.index if splits.get(f) == 'train']
    val_idx = [f for f in features_df.index if splits.get(f) == 'validation']
    test_idx = [f for f in features_df.index if splits.get(f) == 'test']

    X_train = features_df.loc[train_idx].values
    X_val = features_df.loc[val_idx].values
    X_test = features_df.loc[test_idx].values

    y_train = np.array([label_map[f] for f in train_idx])
    y_val = np.array([label_map[f] for f in val_idx])
    y_test = np.array([label_map[f] for f in test_idx])

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


# ============================================================================
# XGBoost
# ============================================================================
def run_xgboost_experiment(X_train, X_val, X_test, y_train, y_val, y_test,
                           experiment_name: str) -> Dict:
    """Run XGBoost experiment with Optuna tuning."""
    print(f"\n  --- {experiment_name}: XGBoost ---")

    # Combine train+val for CV
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Hyperparameter search
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'random_state': RANDOM_STATE,
                'verbosity': 0,
            }
            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring='f1_macro')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        best_params = study.best_trial.params
        print(f"    Best CV F1-macro: {study.best_trial.value:.4f}")
    except ImportError:
        best_params = {
            'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 1.0, 'reg_lambda': 2.0, 'gamma': 0.1,
        }

    # Train final model on train+val
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='mlogloss',
        random_state=RANDOM_STATE, verbosity=0, **best_params
    )
    model.fit(X_trainval, y_trainval)

    # Evaluate on test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = compute_multiclass_metrics(y_test, y_pred, y_prob)
    metrics['best_params'] = best_params

    # Bootstrap CI
    metrics['bootstrap_ci'] = bootstrap_multiclass_ci(y_test, y_pred, y_prob)

    # Save model
    model_path = MODELS_DIR / f'xgboost_{experiment_name}.joblib'
    joblib.dump(model, model_path)

    print(f"    Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Test F1-macro: {metrics['f1_macro']:.4f}")
    print(f"    Test MCC: {metrics['mcc']:.4f}")

    return metrics


# ============================================================================
# FT-Transformer
# ============================================================================
def run_ft_transformer_experiment(X_train, X_val, X_test, y_train, y_val, y_test,
                                  n_features: int, experiment_name: str) -> Dict:
    """Run FT-Transformer experiment."""
    print(f"\n  --- {experiment_name}: FT-Transformer ---")

    # Create datasets
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Build model
    model = FTTransformerMultiClass(
        n_features=n_features, d_token=48, n_layers=3, n_heads=4,
        d_ff=96, dropout=0.3, n_classes=3
    ).to(DEVICE)

    # Class weights from training set
    class_counts = np.bincount(y_train, minlength=3).astype(float)
    class_weights = len(y_train) / (3 * class_counts + 1e-8)
    class_weights = class_weights / class_weights.mean()

    criterion = MultiFocalLoss(num_classes=3, gamma=2.0,
                                class_weights=class_weights)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(200):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                logits = model(X_batch)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(y_batch.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 30:
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    test_preds, test_probs, test_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(y_batch.numpy())

    y_pred = np.array(test_preds)
    y_prob = np.array(test_probs)
    y_true = np.array(test_labels)

    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob)
    metrics['best_val_f1'] = float(best_val_f1)
    metrics['bootstrap_ci'] = bootstrap_multiclass_ci(y_true, y_pred, y_prob)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_f1': best_val_f1,
    }, MODELS_DIR / f'ft_transformer_{experiment_name}.pth')

    print(f"    Best Val F1-macro: {best_val_f1:.4f}")
    print(f"    Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Test F1-macro: {metrics['f1_macro']:.4f}")

    return metrics


# ============================================================================
# Metrics
# ============================================================================
def compute_multiclass_metrics(y_true, y_pred, y_prob) -> Dict:
    """Compute comprehensive multi-class metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1 in enumerate(per_class_f1):
        metrics[f'f1_class_{i}'] = float(f1)

    # AUC (one-vs-rest)
    try:
        metrics['auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
    except ValueError:
        metrics['auc_ovr'] = 0.0

    return metrics


def bootstrap_multiclass_ci(y_true, y_pred, y_prob, n_bootstrap=1000) -> Dict:
    """Bootstrap 95% CI for multi-class metrics."""
    metrics_boot = {'accuracy': [], 'f1_macro': [], 'mcc': []}

    for _ in range(n_bootstrap):
        idx = resample(range(len(y_true)), random_state=None)
        yt = y_true[idx]
        yp = y_pred[idx]

        if len(np.unique(yt)) < 2:
            continue

        metrics_boot['accuracy'].append(accuracy_score(yt, yp))
        metrics_boot['f1_macro'].append(f1_score(yt, yp, average='macro', zero_division=0))
        metrics_boot['mcc'].append(matthews_corrcoef(yt, yp))

    results = {}
    for name, values in metrics_boot.items():
        values = np.array(values)
        results[name] = {
            'mean': float(np.mean(values)),
            'ci_lower': float(np.percentile(values, 2.5)),
            'ci_upper': float(np.percentile(values, 97.5)),
        }
    return results


# ============================================================================
# Visualization
# ============================================================================
def plot_comparison_table(all_results: Dict, save_path: Path):
    """Plot comparison bar chart for all 6 experiments."""
    experiments = list(all_results.keys())
    f1_scores = [all_results[e]['f1_macro'] for e in experiments]
    accuracies = [all_results[e]['accuracy'] for e in experiments]
    mccs = [all_results[e]['mcc'] for e in experiments]

    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar(x, f1_scores, width, label='F1-macro', color='#2ecc71')
    bars3 = ax.bar(x + width, mccs, width, label='MCC', color='#e74c3c')

    ax.set_ylabel('Score')
    ax.set_title('Severity Classification: 6 Experiments Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(all_results: Dict, save_path: Path):
    """Plot confusion matrices for all 6 experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    severity_names = ['Mild', 'Moderate', 'Severe']

    for i, (name, results) in enumerate(all_results.items()):
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=severity_names, yticklabels=severity_names,
                    ax=axes[i], annot_kws={'size': 12})
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.suptitle('Confusion Matrices - All Experiments', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("SEVERITY CLASSIFICATION - 6 EXPERIMENTS")
    print("=" * 60)

    start_time = time.time()
    all_results = {}

    feature_sources = ['gt', 'unet', 'segformer']
    source_labels = {'gt': 'GT', 'unet': 'U-Net', 'segformer': 'SegFormer'}

    for source in feature_sources:
        label = source_labels[source]
        print(f"\n{'='*40}")
        print(f"Feature Source: {label}")
        print(f"{'='*40}")

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
            load_experiment_data(source)

        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"  Train label distribution: {np.bincount(y_train, minlength=3)}")

        # XGBoost experiment
        exp_name = f'{label}_XGB'
        xgb_metrics = run_xgboost_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test, exp_name
        )
        all_results[exp_name] = xgb_metrics

        # FT-Transformer experiment
        exp_name = f'{label}_FTT'
        ftt_metrics = run_ft_transformer_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            n_features=X_train.shape[1], experiment_name=exp_name
        )
        all_results[exp_name] = ftt_metrics

    # Generate comparison outputs
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON OUTPUTS")
    print("=" * 60)

    plot_comparison_table(all_results, FIGURES_DIR / 'fig10_classification_comparison.png')
    plot_confusion_matrices(all_results, FIGURES_DIR / 'fig11_classification_confusion_matrices.png')

    # Save results
    with open(RESULTS_DIR / 'classification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate comparison CSV
    rows = []
    for exp_name, metrics in all_results.items():
        rows.append({
            'Experiment': exp_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1-macro': f"{metrics['f1_macro']:.4f}",
            'F1-weighted': f"{metrics['f1_weighted']:.4f}",
            'MCC': f"{metrics['mcc']:.4f}",
            'AUC (OVR)': f"{metrics.get('auc_ovr', 0):.4f}",
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(RESULTS_DIR / 'classification_comparison.csv', index=False)
    print(f"\nSaved: classification_comparison.csv")
    print(comparison_df.to_string(index=False))

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("SEVERITY CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"\n  Best experiment: {max(all_results, key=lambda k: all_results[k]['f1_macro'])}")


if __name__ == "__main__":
    main()
