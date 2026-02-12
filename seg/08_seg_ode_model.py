"""
08_seg_ode_model.py
====================
SAD ODE Disease Progression Model with Cell-Based Parameterization

Extends src/05_sad_ode_model.py with:
- 3-class severity mapping from classification predictions
- Patient-specific beta from actual bacterial load (continuous)
- 90-day simulations for each severity group
- Sensitivity analysis on gamma, beta, delta

Author: UTI Prediction Team
Date: 2024
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "seg_results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "seg_figures"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ODE parameters (reused from src/05_sad_ode_model.py with cell-based extensions)
ODE_PARAMETERS = {
    'beta_base': {
        'value': 0.15, 'unit': 'day^-1',
        'description': 'Base infection rate',
        'source': '[CITATION NEEDED: UTI incidence rate]',
    },
    'gamma': {
        'value': 0.20, 'unit': 'day^-1',
        'description': 'Progression rate (A -> D)',
        'source': '[CITATION NEEDED: UTI progression]',
    },
    'delta': {
        'value': 0.14, 'unit': 'day^-1',
        'description': 'Recovery rate (D -> S)',
        'source': '[CITATION NEEDED: UTI treatment]',
    },
    'mu': {
        'value': 0.05, 'unit': 'day^-1',
        'description': 'Natural resolution rate (A -> S)',
        'source': '[CITATION NEEDED: Spontaneous resolution]',
    },
    'severity_multiplier_mild': {'value': 0.5, 'unit': 'dimensionless'},
    'severity_multiplier_moderate': {'value': 1.0, 'unit': 'dimensionless'},
    'severity_multiplier_severe': {'value': 2.0, 'unit': 'dimensionless'},
}

SEVERITY_NAMES = {0: 'mild', 1: 'moderate', 2: 'severe'}


def sad_model(y, t, beta, gamma, delta, mu):
    """SAD ODE system: dS/dt, dA/dt, dD/dt."""
    S, A, D = y
    N = S + A + D
    S, A, D = max(0, S), max(0, A), max(0, D)

    dSdt = -beta * S * A / N + delta * D + mu * A
    dAdt = beta * S * A / N - gamma * A - mu * A
    dDdt = gamma * A - delta * D

    return [dSdt, dAdt, dDdt]


def simulate_progression(severity: str, beta_override: float = None,
                         t_max: int = 90) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate disease progression for a severity level."""
    beta_base = ODE_PARAMETERS['beta_base']['value']
    gamma = ODE_PARAMETERS['gamma']['value']
    delta = ODE_PARAMETERS['delta']['value']
    mu = ODE_PARAMETERS['mu']['value']

    mult = ODE_PARAMETERS[f'severity_multiplier_{severity}']['value']
    beta = beta_override if beta_override is not None else beta_base * mult

    y0 = [0.90, 0.10, 0.00]
    t = np.arange(0, t_max, 0.1)
    solution = odeint(sad_model, y0, t, args=(beta, gamma, delta, mu))

    return t, solution


def patient_specific_beta(bacteria_score: float) -> float:
    """
    Map continuous bacterial load score to patient-specific beta.
    Uses a sigmoid-like scaling of base beta.

    bacteria_score: normalized bacterial load [0, 1]
    """
    beta_base = ODE_PARAMETERS['beta_base']['value']
    # Scale from 0.5x to 2.5x based on bacterial load
    multiplier = 0.5 + 2.0 * bacteria_score
    return beta_base * multiplier


def run_scenario_analysis() -> Dict:
    """Run simulations for mild/moderate/severe scenarios."""
    results = {}
    for severity in ['mild', 'moderate', 'severe']:
        t, solution = simulate_progression(severity)
        peak_idx = np.argmax(solution[:, 2])
        results[severity] = {
            'time': t,
            'S': solution[:, 0],
            'A': solution[:, 1],
            'D': solution[:, 2],
            'metrics': {
                'peak_diseased': float(solution[peak_idx, 2]),
                'time_to_peak': float(t[peak_idx]),
                'final_susceptible': float(solution[-1, 0]),
                'total_infected': float(1 - solution[-1, 0]),
            }
        }
    return results


def sensitivity_analysis(param_name: str, param_range: np.ndarray) -> Dict:
    """Sensitivity analysis for a single parameter."""
    results = {'param_values': param_range.tolist(), 'peak_diseased': [], 'time_to_peak': []}
    base_params = {
        'beta': ODE_PARAMETERS['beta_base']['value'],
        'gamma': ODE_PARAMETERS['gamma']['value'],
        'delta': ODE_PARAMETERS['delta']['value'],
        'mu': ODE_PARAMETERS['mu']['value'],
    }

    for value in param_range:
        params = base_params.copy()
        params[param_name] = value
        y0 = [0.90, 0.10, 0.00]
        t = np.arange(0, 90, 0.1)
        solution = odeint(sad_model, y0, t, args=(params['beta'], params['gamma'],
                                                   params['delta'], params['mu']))
        peak_idx = np.argmax(solution[:, 2])
        results['peak_diseased'].append(float(solution[peak_idx, 2]))
        results['time_to_peak'].append(float(t[peak_idx]))

    return results


def apply_to_patient_predictions() -> pd.DataFrame:
    """Apply ODE to individual patient predictions from classification."""
    features_df = pd.read_csv(RESULTS_DIR / 'features_gt.csv', index_col=0)
    labels_df = pd.read_csv(RESULTS_DIR / 'severity_labels.csv')

    with open(RESULTS_DIR / 'classification_results.json', 'r') as f:
        cls_results = json.load(f)

    label_map = dict(zip(labels_df['filename'], labels_df['severity_label']))

    rows = []
    for i, (fname, row) in enumerate(features_df.iterrows()):
        severity_label = label_map.get(fname, 1)
        severity = SEVERITY_NAMES[severity_label]

        # Patient-specific beta from bacterial load
        bacteria_score = min(1.0, max(0.0, row.get('infection_signature', 0.5)))
        beta = patient_specific_beta(abs(bacteria_score))

        t, solution = simulate_progression(severity, beta_override=beta, t_max=90)
        peak_idx = np.argmax(solution[:, 2])

        rows.append({
            'patient_id': i + 1,
            'filename': fname,
            'severity_label': severity_label,
            'severity': severity,
            'patient_beta': beta,
            'peak_diseased': solution[peak_idx, 2],
            'time_to_peak': t[peak_idx],
            'day30_diseased': solution[np.argmin(np.abs(t - 30)), 2],
            'day90_susceptible': solution[-1, 0],
        })

    return pd.DataFrame(rows)


def plot_progression_curves(results: Dict, save_path: Path):
    """Plot SAD progression for all severity levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'S': '#2ecc71', 'A': '#f1c40f', 'D': '#e74c3c'}
    titles = ['Mild Severity', 'Moderate Severity', 'Severe Severity']
    scenarios = ['mild', 'moderate', 'severe']

    for i, (scenario, title) in enumerate(zip(scenarios, titles)):
        t = results[scenario]['time']
        axes[i].fill_between(t, 0, results[scenario]['D'],
                             color=colors['D'], alpha=0.7, label='Diseased')
        axes[i].fill_between(t, results[scenario]['D'],
                             results[scenario]['D'] + results[scenario]['A'],
                             color=colors['A'], alpha=0.7, label='Affected')
        axes[i].fill_between(t, results[scenario]['D'] + results[scenario]['A'],
                             results[scenario]['D'] + results[scenario]['A'] + results[scenario]['S'],
                             color=colors['S'], alpha=0.7, label='Susceptible')
        axes[i].set_xlabel('Time (days)')
        axes[i].set_ylabel('Proportion')
        axes[i].set_title(title)
        axes[i].set_xlim(0, 90)
        axes[i].set_ylim(0, 1)
        axes[i].legend(loc='upper right', fontsize=8)

        m = results[scenario]['metrics']
        axes[i].text(0.95, 0.05, f"Peak D: {m['peak_diseased']:.2%}\nTime: {m['time_to_peak']:.0f}d",
                     transform=axes[i].transAxes, fontsize=8, va='bottom', ha='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('SAD Model: UTI Disease Progression by Severity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sensitivity_heatmap(save_path: Path):
    """Sensitivity analysis plots."""
    params = {
        'beta': ('Infection Rate (beta)', np.linspace(0.05, 0.30, 10)),
        'gamma': ('Progression Rate (gamma)', np.linspace(0.10, 0.40, 10)),
        'delta': ('Recovery Rate (delta)', np.linspace(0.05, 0.25, 10)),
        'mu': ('Natural Resolution (mu)', np.linspace(0.02, 0.15, 10)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (param, (label, values)) in enumerate(params.items()):
        results = sensitivity_analysis(param, values)
        axes[i].plot(values, results['peak_diseased'], 'o-', color='#e74c3c', lw=2, markersize=5)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Peak Diseased Proportion')
        axes[i].set_title(f'Sensitivity to {label}')
        axes[i].grid(True, alpha=0.3)

        baseline = ODE_PARAMETERS[param if param != 'beta' else 'beta_base']['value']
        axes[i].axvline(x=baseline, color='gray', linestyle='--', alpha=0.7, label='Baseline')
        axes[i].legend()

    plt.suptitle('SAD ODE Parameter Sensitivity Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("SAD ODE DISEASE PROGRESSION MODEL")
    print("=" * 60)

    # Step 1: Scenario analysis
    print("\n[Step 1] Running scenario simulations...")
    sim_results = run_scenario_analysis()

    for sev in ['mild', 'moderate', 'severe']:
        m = sim_results[sev]['metrics']
        print(f"  {sev.upper()}: Peak D={m['peak_diseased']:.2%}, "
              f"Time={m['time_to_peak']:.0f}d, Total infected={m['total_infected']:.2%}")

    # Step 2: Patient predictions
    print("\n[Step 2] Applying ODE to individual patients...")
    patient_df = apply_to_patient_predictions()
    patient_df.to_csv(RESULTS_DIR / 'ode_patient_predictions.csv', index=False)

    severity_summary = patient_df.groupby('severity').agg({
        'patient_beta': 'mean',
        'peak_diseased': 'mean',
        'time_to_peak': 'mean',
        'patient_id': 'count',
    }).rename(columns={'patient_id': 'n_patients'})
    print(f"\n{severity_summary}")

    # Step 3: Visualizations
    print("\n[Step 3] Generating visualizations...")
    plot_progression_curves(sim_results, FIGURES_DIR / 'fig16_ode_progression.png')
    plot_sensitivity_heatmap(FIGURES_DIR / 'fig17_ode_sensitivity.png')

    # Step 4: Parameter table
    print("\n[Step 4] Saving parameter table...")
    param_rows = []
    for name, info in ODE_PARAMETERS.items():
        param_rows.append({
            'Parameter': name,
            'Value': info['value'],
            'Unit': info.get('unit', '-'),
            'Description': info.get('description', '-'),
            'Source': info.get('source', 'Model assumption'),
        })
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(RESULTS_DIR / 'ode_parameters.csv', index=False)

    # Step 5: Save outcomes table
    outcome_rows = []
    for sev in ['mild', 'moderate', 'severe']:
        m = sim_results[sev]['metrics']
        outcome_rows.append({
            'Severity': sev.capitalize(),
            'Peak Diseased (%)': f"{m['peak_diseased'] * 100:.1f}",
            'Time to Peak (days)': f"{m['time_to_peak']:.1f}",
            'Final Susceptible (%)': f"{m['final_susceptible'] * 100:.1f}",
            'Total Infected (%)': f"{m['total_infected'] * 100:.1f}",
        })
    outcome_df = pd.DataFrame(outcome_rows)
    outcome_df.to_csv(RESULTS_DIR / 'ode_outcomes.csv', index=False)

    # Save complete results
    ode_results = {
        'model': 'SAD (Susceptible-Affected-Diseased)',
        'parameters': {k: v['value'] for k, v in ODE_PARAMETERS.items()},
        'scenarios': {
            s: {'metrics': d['metrics']} for s, d in sim_results.items()
        },
        'patient_predictions_summary': severity_summary.to_dict(),
    }
    with open(RESULTS_DIR / 'ode_results.json', 'w') as f:
        json.dump(ode_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("SAD ODE MODEL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
