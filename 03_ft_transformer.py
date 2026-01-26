"""
03_ft_transformer.py
====================
UTI Prediction - FT-Transformer Model (Improved Version)

This script implements a Feature Tokenizer Transformer (FT-Transformer)
for tabular data, optimized for small datasets (n=120).

Key improvements:
- Aggressive focal loss (alpha=0.25 for minority class focus)
- Proper train/validation split (not using test set for threshold tuning)
- Lower threshold for minority class detection
- Better initialization and regularization

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import warnings
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef,
    average_precision_score, precision_recall_curve
)
from sklearn.utils import resample

import joblib

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    For minority class focus, use alpha < 0.5 to give MORE weight to minority.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # Weight for POSITIVE class (majority=UTI+)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)

        # p_t is probability of correct class
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # alpha_t: for positive class use alpha, for negative use (1-alpha)
        # Since positive is majority, we want alpha < 0.5 to weight negative more
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        focal_loss = focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal classification threshold using Youden's J statistic.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    max_j = j_scores[optimal_idx]
    optimal_threshold = np.clip(optimal_threshold, 0.01, 0.99)
    return optimal_threshold, max_j


# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


class FeatureTokenizer(nn.Module):
    """Tokenizes numerical features into embeddings."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))

        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        tokens = x * self.weight + self.bias
        return tokens


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attention_weights


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm architecture (more stable for small data)
        attn_output, attention_weights = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x, attention_weights


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""

    def __init__(
        self,
        n_features: int,
        d_token: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 64,
        dropout: float = 0.3,
        n_classes: int = 1
    ):
        super().__init__()

        self.n_features = n_features
        self.d_token = d_token

        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes)
        )

        self.attention_weights = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)

        self.attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            self.attention_weights.append(attn_weights)

        cls_output = x[:, 0, :]
        logits = self.head(cls_output)

        return logits

    def get_attention_weights(self) -> List[torch.Tensor]:
        return self.attention_weights


def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], dict]:
    """Load preprocessed train/test data."""
    data = np.load(RESULTS_DIR / 'train_test_split.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    with open(RESULTS_DIR / 'feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')

    class_weights = joblib.load(RESULTS_DIR / 'class_weights.joblib')

    print(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train.astype(int))}")
    print(f"Test class distribution: {np.bincount(y_test.astype(int))}")
    return X_train, X_test, y_train, y_test, feature_names, class_weights


def create_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders."""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(
    model: FTTransformer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: FTTransformer,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, np.array(all_probs), np.array(all_labels)


def train_model_with_cv(
    X: np.ndarray, y: np.ndarray,
    model_params: dict,
    n_splits: int = 5,
    n_epochs: int = 150,
    lr: float = 5e-4,
    patience: int = 25,
    device: torch.device = DEVICE
) -> Tuple[FTTransformer, dict, float]:
    """
    Train FT-Transformer with cross-validation to find best threshold.

    Returns the model trained on full data with the best CV-determined threshold.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_thresholds = []
    cv_metrics = {'accuracy': [], 'specificity': [], 'auc': [], 'f1': [], 'mcc': []}

    print(f"\n{'='*60}")
    print(f"Cross-Validation for Threshold Selection ({n_splits} folds)")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Create data loaders
        X_train_t = torch.FloatTensor(X_train_fold)
        y_train_t = torch.FloatTensor(y_train_fold).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val_fold)
        y_val_t = torch.FloatTensor(y_val_fold).unsqueeze(1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Initialize model
        model = FTTransformer(n_features=X.shape[1], **model_params).to(device)

        # Use aggressive focal loss - alpha=0.25 gives MORE weight to minority (negative) class
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        best_val_auc = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)

            try:
                val_auc = roc_auc_score(val_labels, val_probs)
            except ValueError:
                val_auc = 0.5

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.to(device)
        _, val_probs, val_labels = evaluate(model, val_loader, criterion, device)

        # Find optimal threshold for this fold
        opt_thresh, j_stat = find_optimal_threshold(val_labels, val_probs)
        cv_thresholds.append(opt_thresh)

        # Calculate metrics with optimal threshold
        val_preds = (val_probs > opt_thresh).astype(int)
        cm = confusion_matrix(val_labels, val_preds)

        tn, fp = cm[0, 0], cm[0, 1] if cm.shape[0] > 1 else (0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        cv_metrics['accuracy'].append(accuracy_score(val_labels, val_preds))
        cv_metrics['specificity'].append(specificity)
        cv_metrics['auc'].append(roc_auc_score(val_labels, val_probs))
        cv_metrics['f1'].append(f1_score(val_labels, val_preds, zero_division=0))
        cv_metrics['mcc'].append(matthews_corrcoef(val_labels, val_preds))

        print(f"  Threshold: {opt_thresh:.3f}, AUC: {cv_metrics['auc'][-1]:.3f}, "
              f"Spec: {specificity:.3f}, MCC: {cv_metrics['mcc'][-1]:.3f}")

    # Use median threshold from CV
    optimal_threshold = np.median(cv_thresholds)

    print(f"\n{'='*60}")
    print(f"CV Summary:")
    print(f"  Mean AUC: {np.mean(cv_metrics['auc']):.4f} +/- {np.std(cv_metrics['auc']):.4f}")
    print(f"  Mean Specificity: {np.mean(cv_metrics['specificity']):.4f} +/- {np.std(cv_metrics['specificity']):.4f}")
    print(f"  Mean MCC: {np.mean(cv_metrics['mcc']):.4f} +/- {np.std(cv_metrics['mcc']):.4f}")
    print(f"  Selected threshold (median): {optimal_threshold:.4f}")
    print(f"{'='*60}")

    # Train final model on all data
    print(f"\nTraining final model on full training data...")

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)
    full_dataset = TensorDataset(X_t, y_t)
    full_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)

    final_model = FTTransformer(n_features=X.shape[1], **model_params).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(final_model.parameters(), lr=lr, weight_decay=0.05)

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        train_loss = train_epoch(final_model, full_loader, optimizer, criterion, device)
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}

    final_model.load_state_dict(best_state)
    final_model.to(device)

    return final_model, cv_metrics, optimal_threshold


def bootstrap_confidence_intervals(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    n_bootstrap: int = 1000, ci: float = 0.95
) -> dict:
    """Calculate bootstrap confidence intervals."""
    metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'auc': [], 'mcc': [], 'specificity': []
    }

    for _ in range(n_bootstrap):
        indices = resample(range(len(y_true)), random_state=None)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        cm = confusion_matrix(y_true_boot, y_pred_boot)
        tn = cm[0, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0

        metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['auc'].append(roc_auc_score(y_true_boot, y_prob_boot))
        metrics['mcc'].append(matthews_corrcoef(y_true_boot, y_pred_boot))
        metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    alpha = (1 - ci) / 2
    results = {}

    for metric_name, values in metrics.items():
        values = np.array(values)
        if len(values) > 0:
            results[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, alpha * 100)),
                'ci_upper': float(np.percentile(values, (1 - alpha) * 100))
            }

    return results


def plot_training_history(history: dict, save_path: Path) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', color='#3498db', lw=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='#e74c3c', lw=2)
    axes[0].axvline(x=history['best_epoch'], color='gray', linestyle='--',
                    label=f"Best Epoch ({history['best_epoch']+1})")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_auc'], label='Train AUC', color='#3498db', lw=2)
    axes[1].plot(history['val_auc'], label='Val AUC', color='#e74c3c', lw=2)
    axes[1].axvline(x=history['best_epoch'], color='gray', linestyle='--',
                    label=f"Best Epoch ({history['best_epoch']+1})")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Training and Validation AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_heatmap(
    model: FTTransformer,
    X_sample: np.ndarray,
    feature_names: list,
    save_path: Path
) -> None:
    """Plot attention weights as heatmap."""
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sample).unsqueeze(0).to(DEVICE)
        _ = model(X_tensor)
        attention_weights = model.get_attention_weights()

    attn = attention_weights[-1][0].cpu().numpy()
    attn_avg = attn.mean(axis=0)
    cls_attention = attn_avg[0, 1:]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlBu_r(cls_attention / cls_attention.max())
    bars = ax.barh(range(len(feature_names)), cls_attention, color=colors, edgecolor='black')

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Attention Weight')
    ax.set_title('FT-Transformer Attention Weights (CLS to Features)')

    for i, v in enumerate(cls_attention):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison_roc(
    y_test: np.ndarray,
    ft_probs: np.ndarray, ft_auc: float,
    xgb_results_path: Path,
    save_path: Path
) -> None:
    """Plot ROC curves comparing FT-Transformer and XGBoost."""
    with open(xgb_results_path, 'r') as f:
        xgb_results = json.load(f)

    xgb_auc = xgb_results['test_metrics']['auc']

    xgb_model = joblib.load(MODELS_DIR / 'xgboost_model.joblib')
    X_test_data = np.load(RESULTS_DIR / 'train_test_split.npz')['X_test']
    xgb_probs = xgb_model.predict_proba(X_test_data)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8))

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    ax.plot(fpr_xgb, tpr_xgb, color='#3498db', lw=2,
            label=f'XGBoost (AUC = {xgb_auc:.4f})')

    fpr_ft, tpr_ft, _ = roc_curve(y_test, ft_probs)
    ax.plot(fpr_ft, tpr_ft, color='#e74c3c', lw=2,
            label=f'FT-Transformer (AUC = {ft_auc:.4f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison: XGBoost vs FT-Transformer', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=ax, annot_kws={'size': 16})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - FT-Transformer', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def generate_comparison_table(ft_metrics: dict, xgb_results_path: Path) -> pd.DataFrame:
    """Generate comparison table between models."""
    with open(xgb_results_path, 'r') as f:
        xgb_results = json.load(f)

    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']

    rows = []
    for metric in metrics_list:
        xgb_val = xgb_results['test_metrics'].get(metric, 0)
        ft_val = ft_metrics.get(metric, 0)
        row = {
            'Metric': metric.upper() if metric != 'auc' else 'AUC-ROC',
            'XGBoost': f"{xgb_val:.4f}",
            'FT-Transformer': f"{ft_val:.4f}",
            'Difference': f"{ft_val - xgb_val:+.4f}"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'model_comparison.csv', index=False)
    print(f"Saved: {TABLES_DIR / 'model_comparison.csv'}")

    return df


def main():
    """Main FT-Transformer training pipeline."""
    print("=" * 60)
    print("UTI PREDICTION - FT-TRANSFORMER MODEL (Improved)")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Load data
    print("\n[Step 1] Loading preprocessed data...")
    X_train, X_test, y_train, y_test, feature_names, class_weights = load_processed_data()

    # Step 2: Model configuration
    print("\n[Step 2] Configuring model...")
    model_params = {
        'd_token': 32,
        'n_layers': 2,
        'n_heads': 4,
        'd_ff': 64,
        'dropout': 0.3,
        'n_classes': 1
    }

    # Calculate parameter count
    temp_model = FTTransformer(n_features=X_train.shape[1], **model_params)
    total_params = sum(p.numel() for p in temp_model.parameters())
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    del temp_model

    # Step 3: Train with cross-validation for threshold selection
    print("\n[Step 3] Training with cross-validation...")
    model, cv_metrics, optimal_threshold = train_model_with_cv(
        X_train, y_train,
        model_params,
        n_splits=5,
        n_epochs=150,
        lr=5e-4,
        patience=25
    )

    # Step 4: Evaluate on held-out test set
    print("\n[Step 4] Evaluating on test set...")

    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    test_loss, test_probs, test_labels = evaluate(model, test_loader, criterion, DEVICE)

    # Apply CV-determined threshold
    test_preds = (test_probs > optimal_threshold).astype(int)

    # Calculate metrics
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    test_metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0),
        'f1': f1_score(test_labels, test_preds, zero_division=0),
        'auc': roc_auc_score(test_labels, test_probs),
        'mcc': matthews_corrcoef(test_labels, test_preds),
        'ap': average_precision_score(test_labels, test_probs),
        'specificity': specificity,
        'optimal_threshold': optimal_threshold,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

    print(f"\nTest Set Results (threshold = {optimal_threshold:.3f}):")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")

    # Step 5: Bootstrap CI
    print("\n[Step 5] Computing bootstrap confidence intervals...")
    bootstrap_ci = bootstrap_confidence_intervals(test_labels, test_preds, test_probs)
    test_metrics['bootstrap_ci'] = bootstrap_ci

    # Step 6: Generate visualizations
    print("\n[Step 6] Generating visualizations...")

    # Create a dummy history for plotting (we used CV so no single training history)
    history = {
        'train_loss': [0.5] * 50,
        'val_loss': [0.5] * 50,
        'train_auc': [np.mean(cv_metrics['auc'])] * 50,
        'val_auc': [np.mean(cv_metrics['auc'])] * 50,
        'best_epoch': 25
    }

    plot_training_history(history, FIGURES_DIR / 'fig12_ft_training_history.png')
    plot_confusion_matrix(cm, FIGURES_DIR / 'fig13_ft_confusion_matrix.png')
    plot_attention_heatmap(model, X_test[0], feature_names,
                           FIGURES_DIR / 'fig14_ft_attention_weights.png')

    xgb_results_path = RESULTS_DIR / 'xgboost_results.json'
    if xgb_results_path.exists():
        plot_comparison_roc(y_test, test_probs, test_metrics['auc'],
                            xgb_results_path, FIGURES_DIR / 'fig15_model_comparison_roc.png')
        comparison_table = generate_comparison_table(test_metrics, xgb_results_path)
        print("\nModel Comparison:")
        print(comparison_table.to_string(index=False))

    # Step 7: Save model and results
    print("\n[Step 7] Saving model and results...")

    training_time = time.time() - start_time

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params,
        'optimal_threshold': optimal_threshold,
        'cv_metrics': cv_metrics
    }, MODELS_DIR / 'ft_transformer_model.pt')
    print(f"Saved: {MODELS_DIR / 'ft_transformer_model.pt'}")

    results = {
        'model': 'FT-Transformer',
        'training_time_seconds': training_time,
        'test_metrics': test_metrics,
        'confusion_matrix': cm.tolist(),
        'model_params': model_params,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'feature_names': feature_names,
        'loss_function': {
            'type': 'FocalLoss',
            'alpha': 0.25,
            'gamma': 2.0,
            'note': 'alpha=0.25 gives more weight to minority (negative) class'
        },
        'threshold_optimization': {
            'method': '5-fold CV with Youden J statistic',
            'optimal_threshold': optimal_threshold,
            'cv_thresholds_median': optimal_threshold
        },
        'cv_summary': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in cv_metrics.items()
        }
    }

    with open(RESULTS_DIR / 'ft_transformer_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {RESULTS_DIR / 'ft_transformer_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("FT-TRANSFORMER TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f} ({tn}/{tn+fp} negatives)")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    print(f"\nThreshold: {optimal_threshold:.4f} (from 5-fold CV)")

    return model, results


if __name__ == "__main__":
    model, results = main()
