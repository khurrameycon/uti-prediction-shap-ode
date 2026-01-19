"""
LaTeX Utilities for UTI Prediction Pipeline
============================================
Functions for generating publication-ready LaTeX tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


def df_to_latex(df: pd.DataFrame,
                caption: str = '',
                label: str = '',
                column_format: str = None,
                escape: bool = False,
                bold_header: bool = True,
                index: bool = False) -> str:
    """
    Convert DataFrame to LaTeX table string.

    Parameters
    ----------
    df : pd.DataFrame
        Data to convert
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    column_format : str, optional
        LaTeX column alignment (e.g., 'lcc')
    escape : bool
        Whether to escape special characters
    bold_header : bool
        Whether to bold column headers
    index : bool
        Whether to include index

    Returns
    -------
    str
        LaTeX table string
    """
    if column_format is None:
        column_format = 'l' + 'c' * len(df.columns)

    # Build header
    if bold_header:
        header = ' & '.join([f'\\textbf{{{col}}}' for col in df.columns])
    else:
        header = ' & '.join(df.columns)

    if index:
        header = f'\\textbf{{{df.index.name or ""}}} & ' + header

    # Build rows
    rows = []
    for idx, row in df.iterrows():
        if escape:
            values = [escape_latex(str(v)) for v in row.values]
        else:
            values = [str(v) for v in row.values]

        if index:
            row_str = f'{idx} & ' + ' & '.join(values)
        else:
            row_str = ' & '.join(values)
        rows.append(row_str + ' \\\\')

    # Assemble table
    latex = [
        '\\begin{table}[htbp]',
        '\\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        f'\\begin{{tabular}}{{{column_format}}}',
        '\\toprule',
        header + ' \\\\',
        '\\midrule',
        '\n'.join(rows),
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ]

    return '\n'.join(latex)


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters.

    Parameters
    ----------
    text : str
        Text to escape

    Returns
    -------
    str
        Escaped text
    """
    special_chars = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}'
    }

    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)

    return text


def format_metric(value: float,
                  ci_lower: float = None,
                  ci_upper: float = None,
                  precision: int = 3) -> str:
    """
    Format metric value with optional confidence interval.

    Parameters
    ----------
    value : float
        Point estimate
    ci_lower : float, optional
        Lower CI bound
    ci_upper : float, optional
        Upper CI bound
    precision : int
        Decimal places

    Returns
    -------
    str
        Formatted string
    """
    if ci_lower is not None and ci_upper is not None:
        return f'{value:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]'
    return f'{value:.{precision}f}'


def create_performance_table(metrics_dict: Dict[str, Dict],
                            model_names: List[str],
                            caption: str = 'Model Performance Comparison',
                            label: str = 'tab:performance') -> str:
    """
    Create LaTeX table comparing model performance.

    Parameters
    ----------
    metrics_dict : dict
        Nested dict: {model_name: {metric_name: value}}
    model_names : list
        Order of models
    caption : str
        Table caption
    label : str
        Table label

    Returns
    -------
    str
        LaTeX table string
    """
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'MCC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'mcc']

    data = []
    for model in model_names:
        row = [model]
        for key in metric_keys:
            if key in metrics_dict.get(model, {}):
                val = metrics_dict[model][key]
                if isinstance(val, dict):
                    row.append(format_metric(val.get('value', val),
                                           val.get('ci_lower'),
                                           val.get('ci_upper')))
                else:
                    row.append(f'{val:.3f}')
            else:
                row.append('--')
        data.append(row)

    df = pd.DataFrame(data, columns=['Model'] + metric_names)
    return df_to_latex(df, caption=caption, label=label)


def create_hyperparameter_table(hyperparams: Dict[str, Dict],
                               caption: str = 'Model Hyperparameters',
                               label: str = 'tab:hyperparams') -> str:
    """
    Create LaTeX table for hyperparameters.

    Parameters
    ----------
    hyperparams : dict
        Nested dict: {model_name: {param_name: value}}
    caption : str
        Table caption
    label : str
        Table label

    Returns
    -------
    str
        LaTeX table string
    """
    rows = []
    for model, params in hyperparams.items():
        for param, value in params.items():
            rows.append({
                'Model': model,
                'Parameter': param,
                'Value': str(value)
            })

    df = pd.DataFrame(rows)
    return df_to_latex(df, caption=caption, label=label, column_format='llr')


def create_feature_importance_table(features: List[str],
                                   importances: List[float],
                                   caption: str = 'Feature Importance',
                                   label: str = 'tab:features') -> str:
    """
    Create LaTeX table for feature importance.

    Parameters
    ----------
    features : list
        Feature names
    importances : list
        Importance values
    caption : str
        Table caption
    label : str
        Table label

    Returns
    -------
    str
        LaTeX table string
    """
    # Sort by importance
    sorted_pairs = sorted(zip(features, importances),
                         key=lambda x: x[1], reverse=True)

    data = []
    for rank, (feat, imp) in enumerate(sorted_pairs, 1):
        data.append({
            'Rank': rank,
            'Feature': feat,
            'Importance': f'{imp:.4f}'
        })

    df = pd.DataFrame(data)
    return df_to_latex(df, caption=caption, label=label, column_format='rlr')


def create_ode_parameter_table(parameters: Dict[str, Dict],
                              caption: str = 'ODE Model Parameters',
                              label: str = 'tab:ode_params') -> str:
    """
    Create LaTeX table for ODE parameters.

    Parameters
    ----------
    parameters : dict
        Dict with parameter info (value, unit, description, source)
    caption : str
        Table caption
    label : str
        Table label

    Returns
    -------
    str
        LaTeX table string
    """
    data = []
    for param, info in parameters.items():
        data.append({
            'Parameter': f'${param}$',
            'Value': info.get('value', '--'),
            'Unit': info.get('unit', '--'),
            'Description': info.get('description', '--'),
            'Source': info.get('source', '[CITATION NEEDED]')
        })

    df = pd.DataFrame(data)
    return df_to_latex(df, caption=caption, label=label,
                      column_format='lllll', escape=False)


def save_latex_table(latex_str: str,
                    output_path: Path,
                    filename: str) -> Path:
    """
    Save LaTeX table to file.

    Parameters
    ----------
    latex_str : str
        LaTeX table string
    output_path : Path
        Output directory
    filename : str
        Filename (without .tex extension)

    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f'{filename}.tex'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex_str)

    return filepath


def create_dataset_characteristics_table(stats: Dict,
                                        caption: str = 'Dataset Characteristics',
                                        label: str = 'tab:dataset') -> str:
    """
    Create LaTeX table for dataset characteristics.

    Parameters
    ----------
    stats : dict
        Dataset statistics
    caption : str
        Table caption
    label : str
        Table label

    Returns
    -------
    str
        LaTeX table string
    """
    data = []

    if 'n_samples' in stats:
        data.append({'Characteristic': 'Total Samples', 'Value': str(stats['n_samples'])})
    if 'n_features' in stats:
        data.append({'Characteristic': 'Number of Features', 'Value': str(stats['n_features'])})
    if 'n_positive' in stats:
        pct = stats.get('pct_positive', stats['n_positive'] / stats['n_samples'] * 100)
        data.append({'Characteristic': 'Positive Cases',
                    'Value': f"{stats['n_positive']} ({pct:.1f}\\%)"})
    if 'n_negative' in stats:
        pct = stats.get('pct_negative', stats['n_negative'] / stats['n_samples'] * 100)
        data.append({'Characteristic': 'Negative Cases',
                    'Value': f"{stats['n_negative']} ({pct:.1f}\\%)"})
    if 'train_size' in stats:
        data.append({'Characteristic': 'Training Samples', 'Value': str(stats['train_size'])})
    if 'test_size' in stats:
        data.append({'Characteristic': 'Test Samples', 'Value': str(stats['test_size'])})

    df = pd.DataFrame(data)
    return df_to_latex(df, caption=caption, label=label, column_format='lr')
