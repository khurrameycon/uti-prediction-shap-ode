"""
01_preprocessing.py
====================
UTI Prediction - Data Preprocessing and Exploratory Data Analysis

This script performs:
1. Data loading and initial inspection
2. Target variable creation (combined UTI indicator)
3. Binary encoding of categorical features
4. Temperature standardization
5. Feature engineering
6. Class distribution analysis
7. Stratified train/test split
8. EDA visualizations
9. Export processed datasets

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories if they don't exist
for dir_path in [FIGURES_DIR, TABLES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load UTI dataset from CSV file.

    Parameters
    ----------
    filepath : Path
        Path to the UTI.csv file

    Returns
    -------
    pd.DataFrame
        Raw dataset
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names for easier handling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with original column names

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names
    """
    # Define mapping for cleaner column names
    column_mapping = {
        'Temperature of patient': 'temperature',
        'Occurrence of nausea': 'nausea',
        'Lumbar pain': 'lumbar_pain',
        'Urine pushing (continuous need for urination)': 'urine_pushing',
        'Micturition pains': 'micturition_pains',
        'Burning of urethra, itch, swelling of urethra outlet': 'burning_urethra',
        'Inflammation of urinary bladder': 'bladder_inflammation',
        'Nephritis of renal pelvis origin': 'nephritis'
    }

    df = df.rename(columns=column_mapping)
    print(f"Cleaned columns: {df.columns.tolist()}")
    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary yes/no features to 0/1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with yes/no string values

    Returns
    -------
    pd.DataFrame
        DataFrame with binary encoded features
    """
    binary_columns = ['nausea', 'lumbar_pain', 'urine_pushing',
                      'micturition_pains', 'burning_urethra',
                      'bladder_inflammation', 'nephritis']

    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})

    print("Binary encoding complete.")
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined UTI target variable.
    UTI positive if either bladder_inflammation OR nephritis is present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with individual condition columns

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'uti' target column
    """
    df['uti'] = ((df['bladder_inflammation'] == 1) | (df['nephritis'] == 1)).astype(int)

    print(f"\nTarget variable 'uti' created:")
    print(f"  - UTI Positive (1): {df['uti'].sum()}")
    print(f"  - UTI Negative (0): {(df['uti'] == 0).sum()}")
    print(f"  - Class ratio: {df['uti'].sum() / len(df):.2%} positive")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for improved prediction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with base features

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features
    """
    # Fever indicator (temperature > 37.5°C is considered fever)
    df['fever'] = (df['temperature'] > 37.5).astype(int)

    # High fever indicator (temperature > 38.5°C)
    df['high_fever'] = (df['temperature'] > 38.5).astype(int)

    # Symptom count (number of symptoms present)
    symptom_cols = ['nausea', 'lumbar_pain', 'urine_pushing',
                    'micturition_pains', 'burning_urethra']
    df['symptom_count'] = df[symptom_cols].sum(axis=1)

    print(f"\nEngineered features added:")
    print(f"  - fever: {df['fever'].sum()} patients with fever (>37.5°C)")
    print(f"  - high_fever: {df['high_fever'].sum()} patients with high fever (>38.5°C)")
    print(f"  - symptom_count: range [{df['symptom_count'].min()}, {df['symptom_count'].max()}]")

    return df


def generate_eda_report(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive EDA statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame

    Returns
    -------
    dict
        Dictionary containing EDA statistics
    """
    report = {}

    # Basic statistics
    report['n_samples'] = len(df)
    report['n_features'] = len(df.columns) - 3  # Exclude targets

    # Target distribution
    report['class_distribution'] = {
        'uti_positive': int(df['uti'].sum()),
        'uti_negative': int((df['uti'] == 0).sum()),
        'positive_ratio': float(df['uti'].mean())
    }

    # Temperature statistics
    report['temperature_stats'] = {
        'mean': float(df['temperature'].mean()),
        'std': float(df['temperature'].std()),
        'min': float(df['temperature'].min()),
        'max': float(df['temperature'].max()),
        'median': float(df['temperature'].median())
    }

    # Feature correlations with target
    feature_cols = ['temperature', 'nausea', 'lumbar_pain', 'urine_pushing',
                    'micturition_pains', 'burning_urethra', 'fever',
                    'high_fever', 'symptom_count']

    correlations = {}
    for col in feature_cols:
        if col in df.columns:
            corr, p_val = stats.pointbiserialr(df['uti'], df[col])
            correlations[col] = {'correlation': float(corr), 'p_value': float(p_val)}

    report['feature_correlations'] = correlations

    # Symptom prevalence
    symptom_cols = ['nausea', 'lumbar_pain', 'urine_pushing',
                    'micturition_pains', 'burning_urethra']
    report['symptom_prevalence'] = {col: float(df[col].mean()) for col in symptom_cols}

    return report


def plot_class_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot class distribution of UTI target variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'uti' column
    save_path : Path
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot
    class_counts = df['uti'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    bars = axes[0].bar(['Negative (0)', 'Positive (1)'], class_counts.values,
                       color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_xlabel('UTI Status')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution (Combined Dataset)')

    # Add count labels on bars
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(count), ha='center', va='bottom', fontweight='bold')

    # Pie chart
    axes[1].pie(class_counts.values, labels=['Negative', 'Positive'],
                autopct='%1.1f%%', colors=colors, explode=(0, 0.05),
                shadow=True, startangle=90)
    axes[1].set_title('Class Proportion')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_temperature_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot temperature distribution by UTI status.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with temperature and uti columns
    save_path : Path
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram
    for uti_status, color, label in [(0, '#2ecc71', 'Negative'), (1, '#e74c3c', 'Positive')]:
        subset = df[df['uti'] == uti_status]['temperature']
        axes[0].hist(subset, bins=15, alpha=0.7, color=color, label=label, edgecolor='black')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Temperature Distribution by UTI Status')
    axes[0].legend()
    axes[0].axvline(x=37.5, color='orange', linestyle='--', label='Fever threshold')

    # Box plot
    df_plot = df[['temperature', 'uti']].copy()
    df_plot['UTI Status'] = df_plot['uti'].map({0: 'Negative', 1: 'Positive'})
    sns.boxplot(data=df_plot, x='UTI Status', y='temperature',
                palette=['#2ecc71', '#e74c3c'], ax=axes[1])
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Temperature Box Plot by UTI Status')
    axes[1].axhline(y=37.5, color='orange', linestyle='--', alpha=0.7)

    # Violin plot
    sns.violinplot(data=df_plot, x='UTI Status', y='temperature',
                   palette=['#2ecc71', '#e74c3c'], ax=axes[2])
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].set_title('Temperature Violin Plot by UTI Status')
    axes[2].axhline(y=37.5, color='orange', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_symptom_prevalence(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot symptom prevalence by UTI status.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with symptom columns
    save_path : Path
        Path to save the figure
    """
    symptom_cols = ['nausea', 'lumbar_pain', 'urine_pushing',
                    'micturition_pains', 'burning_urethra']
    symptom_labels = ['Nausea', 'Lumbar Pain', 'Urine Pushing',
                      'Micturition Pains', 'Burning Urethra']

    # Calculate prevalence by UTI status
    prevalence_negative = [df[df['uti'] == 0][col].mean() * 100 for col in symptom_cols]
    prevalence_positive = [df[df['uti'] == 1][col].mean() * 100 for col in symptom_cols]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(symptom_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, prevalence_negative, width, label='UTI Negative',
                   color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, prevalence_positive, width, label='UTI Positive',
                   color='#e74c3c', edgecolor='black')

    ax.set_xlabel('Symptom')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Symptom Prevalence by UTI Status')
    ax.set_xticks(x)
    ax.set_xticklabels(symptom_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot correlation heatmap of all features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    save_path : Path
        Path to save the figure
    """
    feature_cols = ['temperature', 'nausea', 'lumbar_pain', 'urine_pushing',
                    'micturition_pains', 'burning_urethra', 'fever',
                    'symptom_count', 'uti']

    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})

    ax.set_title('Feature Correlation Heatmap')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance_univariate(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot univariate feature importance based on correlation with target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target
    save_path : Path
        Path to save the figure
    """
    feature_cols = ['temperature', 'nausea', 'lumbar_pain', 'urine_pushing',
                    'micturition_pains', 'burning_urethra', 'fever',
                    'high_fever', 'symptom_count']

    correlations = []
    for col in feature_cols:
        corr, p_val = stats.pointbiserialr(df['uti'], df[col])
        correlations.append({
            'feature': col,
            'correlation': abs(corr),
            'direction': 'positive' if corr > 0 else 'negative',
            'p_value': p_val
        })

    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c' if d == 'positive' else '#3498db'
              for d in corr_df['direction']]

    bars = ax.barh(corr_df['feature'], corr_df['correlation'], color=colors, edgecolor='black')

    ax.set_xlabel('Absolute Correlation with UTI')
    ax.set_ylabel('Feature')
    ax.set_title('Univariate Feature Importance (Point-Biserial Correlation)')
    ax.set_xlim(0, 1)

    # Add significance markers
    for i, (idx, row) in enumerate(corr_df.iterrows()):
        significance = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        ax.text(row['correlation'] + 0.02, i, significance, va='center', fontsize=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Positive correlation'),
                       Patch(facecolor='#3498db', label='Negative correlation')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Prepare stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame
    test_size : float
        Proportion of data for test set

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names)
    """
    # Define feature columns
    feature_cols = ['temperature', 'nausea', 'lumbar_pain', 'urine_pushing',
                    'micturition_pains', 'burning_urethra', 'fever',
                    'high_fever', 'symptom_count']

    X = df[feature_cols].values
    y = df['uti'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\nTrain/Test Split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Training class distribution: {np.bincount(y_train)}")
    print(f"  - Test class distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, feature_cols


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced dataset.

    Parameters
    ----------
    y : np.ndarray
        Target array

    Returns
    -------
    dict
        Class weights dictionary
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))

    class_weights = {}
    for cls in np.unique(y):
        class_weights[cls] = n_samples / (n_classes * np.sum(y == cls))

    print(f"\nClass weights: {class_weights}")
    return class_weights


def save_processed_data(df: pd.DataFrame, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                        feature_cols: list, scaler: StandardScaler) -> None:
    """
    Save all processed data and artifacts.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed DataFrame
    X_train, X_test : np.ndarray
        Feature arrays
    y_train, y_test : np.ndarray
        Target arrays
    feature_cols : list
        Feature column names
    scaler : StandardScaler
        Fitted scaler object
    """
    # Save processed DataFrame
    df.to_csv(RESULTS_DIR / 'processed_data.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'processed_data.csv'}")

    # Save train/test splits
    np.savez(RESULTS_DIR / 'train_test_split.npz',
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)
    print(f"Saved: {RESULTS_DIR / 'train_test_split.npz'}")

    # Save feature names
    with open(RESULTS_DIR / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Saved: {RESULTS_DIR / 'feature_names.txt'}")

    # Save scaler
    joblib.dump(scaler, RESULTS_DIR / 'scaler.joblib')
    print(f"Saved: {RESULTS_DIR / 'scaler.joblib'}")


def generate_dataset_characteristics_table(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Generate dataset characteristics table for manuscript.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame
    report : dict
        EDA report dictionary

    Returns
    -------
    pd.DataFrame
        Characteristics table
    """
    characteristics = []

    # Sample size
    characteristics.append({
        'Characteristic': 'Total Samples',
        'Value': str(report['n_samples']),
        'Notes': 'Small dataset - conservative modeling required'
    })

    # Features
    characteristics.append({
        'Characteristic': 'Number of Features',
        'Value': '9 (6 original + 3 engineered)',
        'Notes': '1 continuous, 8 binary'
    })

    # Class distribution
    pos = report['class_distribution']['uti_positive']
    neg = report['class_distribution']['uti_negative']
    ratio = report['class_distribution']['positive_ratio']
    characteristics.append({
        'Characteristic': 'UTI Positive Cases',
        'Value': f'{pos} ({ratio:.1%})',
        'Notes': 'Combined bladder inflammation and nephritis'
    })

    characteristics.append({
        'Characteristic': 'UTI Negative Cases',
        'Value': f'{neg} ({1-ratio:.1%})',
        'Notes': ''
    })

    # Temperature
    temp_stats = report['temperature_stats']
    characteristics.append({
        'Characteristic': 'Temperature Range',
        'Value': f"{temp_stats['min']:.1f}°C - {temp_stats['max']:.1f}°C",
        'Notes': f"Mean: {temp_stats['mean']:.2f}°C, SD: {temp_stats['std']:.2f}°C"
    })

    char_df = pd.DataFrame(characteristics)

    # Save to CSV
    char_df.to_csv(TABLES_DIR / 'dataset_characteristics.csv', index=False)
    print(f"Saved: {TABLES_DIR / 'dataset_characteristics.csv'}")

    return char_df


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("UTI PREDICTION - DATA PREPROCESSING")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    # Try multiple possible paths
    possible_paths = [
        DATA_DIR / "UTI.csv",
        PROJECT_ROOT / "Dattaset" / "UTI.csv",
    ]

    df = None
    for path in possible_paths:
        if path.exists():
            df = load_data(path)
            break

    if df is None:
        raise FileNotFoundError(f"Could not find UTI.csv in any of: {possible_paths}")

    # Step 2: Clean column names
    print("\n[Step 2] Cleaning column names...")
    df = clean_column_names(df)

    # Step 3: Encode binary features
    print("\n[Step 3] Encoding binary features...")
    df = encode_binary_features(df)

    # Step 4: Create target variable
    print("\n[Step 4] Creating combined UTI target...")
    df = create_target_variable(df)

    # Step 5: Feature engineering
    print("\n[Step 5] Engineering features...")
    df = engineer_features(df)

    # Step 6: Generate EDA report
    print("\n[Step 6] Generating EDA report...")
    eda_report = generate_eda_report(df)

    # Save EDA report
    import json
    with open(RESULTS_DIR / 'eda_report.json', 'w') as f:
        json.dump(eda_report, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'eda_report.json'}")

    # Step 7: Generate visualizations
    print("\n[Step 7] Generating visualizations...")
    plot_class_distribution(df, FIGURES_DIR / 'fig01_class_distribution.png')
    plot_temperature_distribution(df, FIGURES_DIR / 'fig02_temperature_distribution.png')
    plot_symptom_prevalence(df, FIGURES_DIR / 'fig03_symptom_prevalence.png')
    plot_correlation_heatmap(df, FIGURES_DIR / 'fig04_correlation_heatmap.png')
    plot_feature_importance_univariate(df, FIGURES_DIR / 'fig05_univariate_importance.png')

    # Step 8: Train/test split
    print("\n[Step 8] Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)

    # Step 9: Scale features (fit on training only)
    print("\n[Step 9] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 10: Compute class weights
    print("\n[Step 10] Computing class weights...")
    class_weights = compute_class_weights(y_train)

    # Save class weights
    joblib.dump(class_weights, RESULTS_DIR / 'class_weights.joblib')

    # Step 11: Save all processed data
    print("\n[Step 11] Saving processed data...")
    save_processed_data(df, X_train_scaled, X_test_scaled,
                        y_train, y_test, feature_cols, scaler)

    # Step 12: Generate characteristics table
    print("\n[Step 12] Generating dataset characteristics table...")
    char_table = generate_dataset_characteristics_table(df, eda_report)

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nDataset Summary:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - UTI Positive: {df['uti'].sum()} ({df['uti'].mean():.1%})")
    print(f"  - UTI Negative: {(df['uti'] == 0).sum()} ({(df['uti'] == 0).mean():.1%})")
    print(f"\nTrain/Test Split (80/20):")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Test: {len(X_test)} samples")
    print(f"\nOutputs saved to:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Tables: {TABLES_DIR}")
    print(f"  - Results: {RESULTS_DIR}")

    return df, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


if __name__ == "__main__":
    df, X_train, X_test, y_train, y_test, feature_cols, scaler = main()
