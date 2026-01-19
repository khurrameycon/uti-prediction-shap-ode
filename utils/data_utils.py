"""
Data Utilities for UTI Prediction Pipeline
===========================================
Functions for data loading, preprocessing, and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_uti_dataset(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load UTI dataset from CSV file.

    Parameters
    ----------
    data_path : str or Path
        Path to UTI.csv file

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(data_path)
    return df


def encode_binary_features(df: pd.DataFrame,
                          binary_columns: List[str],
                          yes_value: str = 'yes',
                          no_value: str = 'no') -> pd.DataFrame:
    """
    Encode binary yes/no columns to 1/0.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    binary_columns : list
        Columns to encode
    yes_value : str
        Value representing 'yes' (default: 'yes')
    no_value : str
        Value representing 'no' (default: 'no')

    Returns
    -------
    pd.DataFrame
        Dataframe with encoded columns
    """
    df_encoded = df.copy()
    for col in binary_columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({yes_value: 1, no_value: 0})
    return df_encoded


def create_combined_target(df: pd.DataFrame,
                          target_cols: List[str],
                          target_name: str = 'uti') -> pd.DataFrame:
    """
    Create combined target variable from multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_cols : list
        Columns to combine (OR logic)
    target_name : str
        Name for combined target column

    Returns
    -------
    pd.DataFrame
        Dataframe with new target column
    """
    df_new = df.copy()
    df_new[target_name] = (df[target_cols].max(axis=1) > 0).astype(int)
    return df_new


def engineer_features(df: pd.DataFrame,
                     symptom_columns: List[str],
                     temp_column: str = 'Temperature') -> pd.DataFrame:
    """
    Create engineered features for UTI prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    symptom_columns : list
        Binary symptom columns to sum
    temp_column : str
        Temperature column name

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    df_new = df.copy()

    # Symptom count
    df_new['symptom_count'] = df[symptom_columns].sum(axis=1)

    # Fever indicator (temperature > 37.5°C)
    if temp_column in df.columns:
        df_new['has_fever'] = (df[temp_column] > 37.5).astype(int)

    return df_new


def stratified_split(X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple:
    """
    Perform stratified train-test split.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Test set proportion
    random_state : int
        Random seed

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size,
                           stratify=y, random_state=random_state)


def get_cv_splitter(n_splits: int = 5,
                   shuffle: bool = True,
                   random_state: int = 42) -> StratifiedKFold:
    """
    Get stratified K-fold cross-validator.

    Parameters
    ----------
    n_splits : int
        Number of folds
    shuffle : bool
        Whether to shuffle
    random_state : int
        Random seed

    Returns
    -------
    StratifiedKFold
        Cross-validator object
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                          random_state=random_state)


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights.

    Parameters
    ----------
    y : np.ndarray
        Target array

    Returns
    -------
    dict
        Class weights dictionary
    """
    classes = np.unique(y)
    n_samples = len(y)
    weights = {}
    for c in classes:
        weights[c] = n_samples / (len(classes) * np.sum(y == c))
    return weights


def validate_dataset(df: pd.DataFrame,
                    required_columns: List[str]) -> Dict[str, any]:
    """
    Validate dataset structure and quality.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate
    required_columns : list
        Expected columns

    Returns
    -------
    dict
        Validation results
    """
    results = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'missing_columns': [c for c in required_columns if c not in df.columns],
        'null_counts': df.isnull().sum().to_dict(),
        'has_nulls': df.isnull().any().any(),
        'is_valid': True
    }

    if results['missing_columns']:
        results['is_valid'] = False

    return results
