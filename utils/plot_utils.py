"""
Plotting Utilities for UTI Prediction Pipeline
===============================================
Functions for creating publication-quality visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


# Publication settings
PUBLICATION_SETTINGS = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
}

# Colorblind-friendly palette
COLORBLIND_PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'yellow': '#F0E442',
    'black': '#000000'
}


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update(PUBLICATION_SETTINGS)
    sns.set_style("whitegrid")
    sns.set_context("paper")


def save_figure(fig: plt.Figure,
                output_path: Path,
                filename: str,
                formats: List[str] = ['png', 'pdf'],
                dpi: int = 300) -> List[Path]:
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure
    output_path : Path
        Output directory
    filename : str
        Filename without extension
    formats : list
        Output formats
    dpi : int
        Resolution

    Returns
    -------
    list
        Paths to saved files
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        saved_paths.append(filepath)

    return saved_paths


def plot_class_distribution(y: np.ndarray,
                           class_names: List[str] = ['Negative', 'Positive'],
                           title: str = 'Class Distribution',
                           colors: Optional[List[str]] = None) -> plt.Figure:
    """
    Create class distribution bar plot.

    Parameters
    ----------
    y : np.ndarray
        Target array
    class_names : list
        Names for each class
    title : str
        Plot title
    colors : list, optional
        Bar colors

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if colors is None:
        colors = [COLORBLIND_PALETTE['blue'], COLORBLIND_PALETTE['orange']]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    unique, counts = np.unique(y, return_counts=True)
    bars = ax.bar(class_names[:len(unique)], counts, color=colors[:len(unique)])

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{count}\n({count/len(y)*100:.1f}%)',
               ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_ylim(0, max(counts) * 1.2)

    plt.tight_layout()
    return fig


def plot_roc_curve(fpr: np.ndarray,
                   tpr: np.ndarray,
                   auc_score: float,
                   model_name: str = 'Model',
                   color: str = None) -> plt.Figure:
    """
    Create ROC curve plot.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    auc_score : float
        AUC score
    model_name : str
        Model name for legend
    color : str, optional
        Line color

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if color is None:
        color = COLORBLIND_PALETTE['blue']

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color=color, lw=2,
           label=f'{model_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str] = ['Negative', 'Positive'],
                         title: str = 'Confusion Matrix',
                         cmap: str = 'Blues',
                         normalize: bool = False) -> plt.Figure:
    """
    Create confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (2x2)
    class_names : list
        Class labels
    title : str
        Plot title
    cmap : str
        Colormap name
    normalize : bool
        Whether to normalize values

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=True)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_feature_importance(features: List[str],
                           importances: np.ndarray,
                           title: str = 'Feature Importance',
                           color: str = None,
                           top_n: int = None) -> plt.Figure:
    """
    Create horizontal bar plot for feature importance.

    Parameters
    ----------
    features : list
        Feature names
    importances : np.ndarray
        Importance values
    title : str
        Plot title
    color : str, optional
        Bar color
    top_n : int, optional
        Show only top N features

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if color is None:
        color = COLORBLIND_PALETTE['blue']

    # Sort by importance
    indices = np.argsort(importances)
    if top_n:
        indices = indices[-top_n:]

    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_features) * 0.4)))

    ax.barh(range(len(sorted_features)), sorted_importances, color=color)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Importance')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_training_history(history: Dict[str, List[float]],
                         title: str = 'Training History') -> plt.Figure:
    """
    Plot training and validation loss/metrics over epochs.

    Parameters
    ----------
    history : dict
        Dictionary with 'train_loss', 'val_loss', etc.
    title : str
        Plot title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_publication_style()

    n_plots = 0
    if 'train_loss' in history:
        n_plots += 1
    if 'train_acc' in history:
        n_plots += 1

    n_plots = max(1, n_plots)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if 'train_loss' in history:
        ax = axes[plot_idx]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'],
               color=COLORBLIND_PALETTE['blue'], label='Train')
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'],
                   color=COLORBLIND_PALETTE['orange'], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        plot_idx += 1

    if 'train_acc' in history:
        ax = axes[plot_idx]
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'],
               color=COLORBLIND_PALETTE['blue'], label='Train')
        if 'val_acc' in history:
            ax.plot(epochs, history['val_acc'],
                   color=COLORBLIND_PALETTE['orange'], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy')
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_model_comparison(models: List[str],
                         metrics: Dict[str, List[float]],
                         title: str = 'Model Comparison') -> plt.Figure:
    """
    Create grouped bar plot comparing multiple models.

    Parameters
    ----------
    models : list
        Model names
    metrics : dict
        Dictionary mapping metric names to lists of values
    title : str
        Plot title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_publication_style()

    x = np.arange(len(models))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = list(COLORBLIND_PALETTE.values())

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * i - width * (len(metrics) - 1) / 2
        bars = ax.bar(x + offset, values, width,
                     label=metric_name, color=colors[i % len(colors)])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    return fig


def create_multipanel_figure(n_rows: int,
                             n_cols: int,
                             figsize: Tuple[float, float] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a multi-panel figure layout.

    Parameters
    ----------
    n_rows : int
        Number of rows
    n_cols : int
        Number of columns
    figsize : tuple, optional
        Figure size (width, height)

    Returns
    -------
    tuple
        (Figure, array of axes)
    """
    set_publication_style()

    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Add panel labels (a, b, c, ...)
    if n_rows * n_cols > 1:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, ax in enumerate(axes_flat):
            ax.text(-0.1, 1.1, chr(97 + i), transform=ax.transAxes,
                   fontsize=12, fontweight='bold', va='top')

    return fig, axes
