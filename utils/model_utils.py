"""
Model Utilities for UTI Prediction Pipeline
============================================
Functions for model training, evaluation, and metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from scipy import stats
import json
from pathlib import Path


def compute_classification_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Predicted probabilities for positive class

    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = np.nan
            metrics['avg_precision'] = np.nan

    return metrics


def bootstrap_confidence_interval(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  metric_func: Callable,
                                  n_bootstrap: int = 1000,
                                  confidence: float = 0.95,
                                  random_state: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    metric_func : callable
        Sklearn metric function
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    random_state : int
        Random seed

    Returns
    -------
    tuple
        (point_estimate, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    point_estimate = metric_func(y_true, y_pred)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return point_estimate, lower, upper


def compute_metrics_with_ci(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None,
                           n_bootstrap: int = 1000) -> Dict[str, Dict]:
    """
    Compute all metrics with bootstrap confidence intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Predicted probabilities
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    dict
        Metrics with point estimates and CIs
    """
    metric_funcs = {
        'accuracy': accuracy_score,
        'precision': lambda y, p: precision_score(y, p, zero_division=0),
        'recall': lambda y, p: recall_score(y, p, zero_division=0),
        'f1': lambda y, p: f1_score(y, p, zero_division=0),
        'mcc': matthews_corrcoef
    }

    results = {}
    for name, func in metric_funcs.items():
        point, lower, upper = bootstrap_confidence_interval(
            y_true, y_pred, func, n_bootstrap
        )
        results[name] = {
            'value': point,
            'ci_lower': lower,
            'ci_upper': upper
        }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            results['auc_roc'] = {'value': auc, 'ci_lower': np.nan, 'ci_upper': np.nan}
        except ValueError:
            pass

    return results


def get_confusion_matrix_dict(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict[str, int]:
    """
    Get confusion matrix as a dictionary.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    dict
        Confusion matrix components
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp)
    }


def get_roc_curve_data(y_true: np.ndarray,
                       y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get ROC curve data points.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities

    Returns
    -------
    dict
        FPR, TPR, and thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }


def get_pr_curve_data(y_true: np.ndarray,
                      y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get Precision-Recall curve data points.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities

    Returns
    -------
    dict
        Precision, recall, and thresholds
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist()
    }


def save_model_results(results: Dict,
                       output_path: Path,
                       filename: str) -> None:
    """
    Save model results to JSON file.

    Parameters
    ----------
    results : dict
        Results dictionary
    output_path : Path
        Output directory
    filename : str
        Output filename (without extension)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_serializable = convert_numpy(results)

    with open(output_path / f"{filename}.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)


def permutation_test(y_true: np.ndarray,
                     y_pred_a: np.ndarray,
                     y_pred_b: np.ndarray,
                     metric_func: Callable,
                     n_permutations: int = 10000,
                     random_state: int = 42) -> Tuple[float, float]:
    """
    Perform permutation test to compare two models.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_a : np.ndarray
        Predictions from model A
    y_pred_b : np.ndarray
        Predictions from model B
    metric_func : callable
        Metric function to compare
    n_permutations : int
        Number of permutations
    random_state : int
        Random seed

    Returns
    -------
    tuple
        (observed_difference, p_value)
    """
    rng = np.random.RandomState(random_state)

    score_a = metric_func(y_true, y_pred_a)
    score_b = metric_func(y_true, y_pred_b)
    observed_diff = score_a - score_b

    combined = np.column_stack([y_pred_a, y_pred_b])
    perm_diffs = []

    for _ in range(n_permutations):
        # Randomly swap predictions
        swap_mask = rng.randint(0, 2, len(y_true))
        perm_a = np.where(swap_mask == 0, combined[:, 0], combined[:, 1])
        perm_b = np.where(swap_mask == 0, combined[:, 1], combined[:, 0])

        perm_diff = metric_func(y_true, perm_a) - metric_func(y_true, perm_b)
        perm_diffs.append(perm_diff)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value
