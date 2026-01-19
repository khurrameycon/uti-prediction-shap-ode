"""
07_patient_probability_output.py
================================
UTI Prediction - Per-Patient Probability Output

This script generates per-patient probability outputs with 4 severity classes
as requested for mathematical analysis and ODE modeling.

Output Format (Excel):
- Patient_ID: Unique identifier for each patient
- True_Label: Original ground truth label
- Predicted_Label: Predicted class label
- True_Class_Index: Numerical index of true class
- Predicted_Class_Index: Numerical index of predicted class
- Prob_No_Impairment: Probability of no UTI impairment
- Prob_Very_Mild_Impairment: Probability of very mild impairment
- Prob_Mild_Impairment: Probability of mild impairment
- Prob_Moderate_Impairment: Probability of moderate/severe impairment

Severity Mapping (from binary UTI probability):
- No Impairment: UTI prob < 0.25
- Very Mild Impairment: 0.25 <= UTI prob < 0.50
- Mild Impairment: 0.50 <= UTI prob < 0.75
- Moderate Impairment: UTI prob >= 0.75

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import xgboost as xgb
import joblib
from scipy.special import softmax

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DATA_DIR = PROJECT_ROOT / "Dataset"

# Device configuration for FT-Transformer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Severity class definitions
SEVERITY_CLASSES = ['No_Impairment', 'Very_Mild_Impairment', 'Mild_Impairment', 'Moderate_Impairment']
SEVERITY_THRESHOLDS = [0.25, 0.50, 0.75]  # Thresholds between classes


class FeatureTokenizer(nn.Module):
    """Tokenizes numerical features into embeddings."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in = d_token
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

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
        attention_weights = torch.softmax(scores, dim=-1)
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
        attn_output, attention_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
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
        dropout: float = 0.4,
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


def binary_to_severity_probabilities(binary_prob: float, temperature: float = 0.15) -> np.ndarray:
    """
    Convert binary UTI probability to 4-class severity probabilities.

    Uses a soft-thresholding approach with temperature scaling to generate
    probability distributions across severity classes with clear dominant class.

    Parameters
    ----------
    binary_prob : float
        Binary UTI probability (0-1)
    temperature : float
        Softmax temperature for smoothing (lower = sharper peaks)
        Default 0.15 produces split like [0.1, 0.2, 0.6, 0.1]

    Returns
    -------
    np.ndarray
        4-element array of severity probabilities (sums to 1)
    """
    # Create distance-based logits for each severity class
    # Centers of each severity class
    class_centers = [0.125, 0.375, 0.625, 0.875]  # Centers for 4 classes

    # Calculate negative squared distance to each class center
    logits = np.array([-((binary_prob - center) ** 2) for center in class_centers])

    # Apply temperature scaling and softmax
    # Lower temperature = more peaked distribution (one dominant probability)
    logits = logits / temperature
    probs = softmax(logits)

    return probs


def get_severity_class(binary_prob: float) -> Tuple[int, str]:
    """
    Get severity class from binary probability.

    Parameters
    ----------
    binary_prob : float
        Binary UTI probability

    Returns
    -------
    tuple
        (class_index, class_name)
    """
    if binary_prob < 0.25:
        return 0, 'No_Impairment'
    elif binary_prob < 0.50:
        return 1, 'Very_Mild_Impairment'
    elif binary_prob < 0.75:
        return 2, 'Mild_Impairment'
    else:
        return 3, 'Moderate_Impairment'


def load_xgboost_model() -> xgb.XGBClassifier:
    """Load trained XGBoost model."""
    model_path = MODELS_DIR / 'xgboost_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {model_path}. Run 02_xgboost_baseline.py first.")
    return joblib.load(model_path)


def load_ft_transformer_model(n_features: int) -> FTTransformer:
    """Load trained FT-Transformer model."""
    model_path = MODELS_DIR / 'ft_transformer_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"FT-Transformer model not found at {model_path}. Run 03_ft_transformer.py first.")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_params = checkpoint['model_params']

    model = FTTransformer(n_features=n_features, **model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    return model


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load preprocessed train/test data."""
    data = np.load(RESULTS_DIR / 'train_test_split.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    with open(RESULTS_DIR / 'feature_names.txt', 'r') as f:
        feature_names = f.read().strip().split('\n')

    return X_train, X_test, y_train, y_test, feature_names


def generate_patient_probabilities_xgboost(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    dataset_name: str = 'test'
) -> pd.DataFrame:
    """
    Generate per-patient probability outputs for XGBoost model.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained XGBoost model
    X : np.ndarray
        Feature matrix
    y_true : np.ndarray
        True labels
    dataset_name : str
        Name prefix for patient IDs

    Returns
    -------
    pd.DataFrame
        Patient-level probability output
    """
    # Get binary probabilities
    binary_probs = model.predict_proba(X)[:, 1]

    records = []
    for i in range(len(X)):
        binary_prob = binary_probs[i]
        true_label = int(y_true[i])

        # Get predicted severity class
        pred_class_idx, pred_class_name = get_severity_class(binary_prob)

        # Get true severity class (based on original binary label)
        # If UTI positive (1), map to Mild/Moderate; if negative (0), map to No/Very_Mild
        if true_label == 1:
            true_class_idx = 3  # Moderate_Impairment (has UTI)
            true_class_name = 'Moderate_Impairment'
        else:
            true_class_idx = 0  # No_Impairment (no UTI)
            true_class_name = 'No_Impairment'

        # Generate 4-class probabilities
        severity_probs = binary_to_severity_probabilities(binary_prob)

        record = {
            'Patient_ID': f'{dataset_name}_patient_{i+1:03d}',
            'True_Label': true_class_name,
            'Predicted_Label': pred_class_name,
            'True_Class_Index': true_class_idx,
            'Predicted_Class_Index': pred_class_idx,
            'Prob_No_Impairment': severity_probs[0],
            'Prob_Very_Mild_Impairment': severity_probs[1],
            'Prob_Mild_Impairment': severity_probs[2],
            'Prob_Moderate_Impairment': severity_probs[3],
            'Binary_UTI_Probability': binary_prob
        }
        records.append(record)

    return pd.DataFrame(records)


def generate_patient_probabilities_ft_transformer(
    model: FTTransformer,
    X: np.ndarray,
    y_true: np.ndarray,
    dataset_name: str = 'test'
) -> pd.DataFrame:
    """
    Generate per-patient probability outputs for FT-Transformer model.

    Parameters
    ----------
    model : FTTransformer
        Trained FT-Transformer model
    X : np.ndarray
        Feature matrix
    y_true : np.ndarray
        True labels
    dataset_name : str
        Name prefix for patient IDs

    Returns
    -------
    pd.DataFrame
        Patient-level probability output
    """
    model.eval()

    # Get binary probabilities
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        logits = model(X_tensor)
        binary_probs = torch.sigmoid(logits).cpu().numpy().flatten()

    records = []
    for i in range(len(X)):
        binary_prob = binary_probs[i]
        true_label = int(y_true[i])

        # Get predicted severity class
        pred_class_idx, pred_class_name = get_severity_class(binary_prob)

        # Get true severity class
        if true_label == 1:
            true_class_idx = 3
            true_class_name = 'Moderate_Impairment'
        else:
            true_class_idx = 0
            true_class_name = 'No_Impairment'

        # Generate 4-class probabilities
        severity_probs = binary_to_severity_probabilities(binary_prob)

        record = {
            'Patient_ID': f'{dataset_name}_patient_{i+1:03d}',
            'True_Label': true_class_name,
            'Predicted_Label': pred_class_name,
            'True_Class_Index': true_class_idx,
            'Predicted_Class_Index': pred_class_idx,
            'Prob_No_Impairment': severity_probs[0],
            'Prob_Very_Mild_Impairment': severity_probs[1],
            'Prob_Mild_Impairment': severity_probs[2],
            'Prob_Moderate_Impairment': severity_probs[3],
            'Binary_UTI_Probability': binary_prob
        }
        records.append(record)

    return pd.DataFrame(records)


def main():
    """Main function to generate per-patient probability outputs."""
    print("=" * 70)
    print("UTI PREDICTION - PER-PATIENT PROBABILITY OUTPUT GENERATOR")
    print("=" * 70)

    # Load data
    print("\n[Step 1] Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Combine all data for full dataset output
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Total samples: {len(X_all)}")

    # Create output directory
    output_dir = PROJECT_ROOT / "outputs" / "patient_probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # XGBoost Outputs
    # ============================================
    print("\n[Step 2] Generating XGBoost predictions...")
    try:
        xgb_model = load_xgboost_model()

        # Test set
        df_xgb_test = generate_patient_probabilities_xgboost(xgb_model, X_test, y_test, 'test')
        df_xgb_test.to_excel(output_dir / 'XGBoost_Test_Patient_Probabilities.xlsx', index=False)
        print(f"  Saved: XGBoost_Test_Patient_Probabilities.xlsx ({len(df_xgb_test)} patients)")

        # Training set
        df_xgb_train = generate_patient_probabilities_xgboost(xgb_model, X_train, y_train, 'train')
        df_xgb_train.to_excel(output_dir / 'XGBoost_Train_Patient_Probabilities.xlsx', index=False)
        print(f"  Saved: XGBoost_Train_Patient_Probabilities.xlsx ({len(df_xgb_train)} patients)")

        # Full dataset
        df_xgb_all = generate_patient_probabilities_xgboost(xgb_model, X_all, y_all, 'all')
        df_xgb_all.to_excel(output_dir / 'XGBoost_All_Patient_Probabilities.xlsx', index=False)
        print(f"  Saved: XGBoost_All_Patient_Probabilities.xlsx ({len(df_xgb_all)} patients)")

    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        df_xgb_test = None

    # ============================================
    # FT-Transformer Outputs
    # ============================================
    print("\n[Step 3] Generating FT-Transformer predictions...")
    try:
        ft_model = load_ft_transformer_model(n_features=X_test.shape[1])

        # Test set
        df_ft_test = generate_patient_probabilities_ft_transformer(ft_model, X_test, y_test, 'test')
        df_ft_test.to_excel(output_dir / 'FT_Transformer_Test_Patient_Probabilities.xlsx', index=False)
        print(f"  Saved: FT_Transformer_Test_Patient_Probabilities.xlsx ({len(df_ft_test)} patients)")

        # Training set
        df_ft_train = generate_patient_probabilities_ft_transformer(ft_model, X_train, y_train, 'train')
        df_ft_train.to_excel(output_dir / 'FT_Transformer_Train_Patient_Probabilities.xlsx', index=False)
        print(f"  Saved: FT_Transformer_Train_Patient_Probabilities.xlsx ({len(df_ft_train)} patients)")

        # Full dataset
        df_ft_all = generate_patient_probabilities_ft_transformer(ft_model, X_all, y_all, 'all')
        df_ft_all.to_excel(output_dir / 'FT_Transformer_All_Patient_Probabilities.xlsx', index=False)
        print(f"  Saved: FT_Transformer_All_Patient_Probabilities.xlsx ({len(df_ft_all)} patients)")

    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        df_ft_test = None

    # ============================================
    # Summary Statistics
    # ============================================
    print("\n[Step 4] Generating summary statistics...")

    summary_data = []

    if df_xgb_test is not None:
        xgb_severity_dist = df_xgb_test['Predicted_Label'].value_counts()
        summary_data.append({
            'Model': 'XGBoost',
            'Dataset': 'Test',
            'N_Patients': len(df_xgb_test),
            'No_Impairment': xgb_severity_dist.get('No_Impairment', 0),
            'Very_Mild_Impairment': xgb_severity_dist.get('Very_Mild_Impairment', 0),
            'Mild_Impairment': xgb_severity_dist.get('Mild_Impairment', 0),
            'Moderate_Impairment': xgb_severity_dist.get('Moderate_Impairment', 0)
        })

    if df_ft_test is not None:
        ft_severity_dist = df_ft_test['Predicted_Label'].value_counts()
        summary_data.append({
            'Model': 'FT-Transformer',
            'Dataset': 'Test',
            'N_Patients': len(df_ft_test),
            'No_Impairment': ft_severity_dist.get('No_Impairment', 0),
            'Very_Mild_Impairment': ft_severity_dist.get('Very_Mild_Impairment', 0),
            'Mild_Impairment': ft_severity_dist.get('Mild_Impairment', 0),
            'Moderate_Impairment': ft_severity_dist.get('Moderate_Impairment', 0)
        })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(output_dir / 'Severity_Distribution_Summary.xlsx', index=False)
        print(f"  Saved: Severity_Distribution_Summary.xlsx")

        print("\n" + "=" * 70)
        print("SEVERITY DISTRIBUTION SUMMARY")
        print("=" * 70)
        print(df_summary.to_string(index=False))

    # ============================================
    # Print Example Output
    # ============================================
    print("\n" + "=" * 70)
    print("EXAMPLE OUTPUT (First 5 patients from XGBoost Test Set)")
    print("=" * 70)

    if df_xgb_test is not None:
        display_cols = [
            'Patient_ID', 'True_Label', 'Predicted_Label',
            'True_Class_Index', 'Predicted_Class_Index',
            'Prob_No_Impairment', 'Prob_Very_Mild_Impairment',
            'Prob_Mild_Impairment', 'Prob_Moderate_Impairment'
        ]
        print(df_xgb_test[display_cols].head().to_string(index=False))

        print("\n" + "-" * 70)
        print("Example argmax interpretation:")
        print("-" * 70)
        for i in range(min(3, len(df_xgb_test))):
            row = df_xgb_test.iloc[i]
            probs = [
                row['Prob_No_Impairment'],
                row['Prob_Very_Mild_Impairment'],
                row['Prob_Mild_Impairment'],
                row['Prob_Moderate_Impairment']
            ]
            print(f"Patient {row['Patient_ID']}:")
            print(f"  Probabilities: [{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}, {probs[3]:.2f}]")
            print(f"  argmax selects index {np.argmax(probs)} -> {SEVERITY_CLASSES[np.argmax(probs)]}")
            print()

    print("\n" + "=" * 70)
    print("OUTPUT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.glob('*.xlsx'):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
