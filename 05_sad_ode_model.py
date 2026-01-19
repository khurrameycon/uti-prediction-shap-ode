"""
05_sad_ode_model.py
===================
UTI Prediction - SAD ODE Disease Progression Model

Implements a Susceptible-Affected-Diseased (SAD) compartmental model
for UTI disease progression based on predicted severity levels.

Model States:
- S (Susceptible): At-risk individuals
- A (Affected): Early/prodromal UTI stage
- D (Diseased): Symptomatic UTI

The model uses ML-predicted UTI probability to inform disease
progression rates, enabling personalized outcome simulation.

Author: UTI Prediction Team
Date: 2024

NOTE: ODE parameters require literature verification.
      See References/citations_log.md for verification status.
"""

import os
import sys
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
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

# Plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# ==============================================================================
# ODE PARAMETER DEFINITIONS
# ==============================================================================
# NOTE: These parameters are based on UTI epidemiology literature.
# Each parameter includes a citation placeholder that must be verified.
# See References/citations_log.md for verification status.

ODE_PARAMETERS = {
    'beta_base': {
        'value': 0.15,
        'description': 'Base transmission/infection rate (per day)',
        'unit': 'day^-1',
        'source': '[CITATION NEEDED: UTI incidence rate]',
        'notes': 'Rate at which susceptible individuals become affected'
    },
    'gamma': {
        'value': 0.20,
        'description': 'Progression rate from Affected to Diseased',
        'unit': 'day^-1',
        'source': '[CITATION NEEDED: UTI progression studies]',
        'notes': 'Estimated mean time to symptomatic disease: ~5 days'
    },
    'delta': {
        'value': 0.14,
        'description': 'Recovery rate from Diseased state',
        'unit': 'day^-1',
        'source': '[CITATION NEEDED: UTI treatment outcomes]',
        'notes': 'With treatment, recovery in ~7 days; estimated as 1/7'
    },
    'mu': {
        'value': 0.05,
        'description': 'Natural recovery rate from Affected (mild resolution)',
        'unit': 'day^-1',
        'source': '[CITATION NEEDED: Spontaneous UTI resolution]',
        'notes': 'Some mild UTIs resolve without treatment (~20 days mean)'
    },
    'severity_multiplier_mild': {
        'value': 0.5,
        'description': 'Severity multiplier for mild cases (P(UTI) < 0.4)',
        'unit': 'dimensionless',
        'source': 'Model assumption based on severity stratification',
        'notes': 'Reduces progression rate for low-risk individuals'
    },
    'severity_multiplier_moderate': {
        'value': 1.0,
        'description': 'Severity multiplier for moderate cases (0.4 <= P(UTI) < 0.7)',
        'unit': 'dimensionless',
        'source': 'Model assumption based on severity stratification',
        'notes': 'Baseline progression rate'
    },
    'severity_multiplier_severe': {
        'value': 2.0,
        'description': 'Severity multiplier for severe cases (P(UTI) >= 0.7)',
        'unit': 'dimensionless',
        'source': 'Model assumption based on severity stratification',
        'notes': 'Increases progression rate for high-risk individuals'
    }
}


def get_severity_category(probability: float) -> Tuple[str, float]:
    """
    Map predicted UTI probability to severity category and multiplier.

    Parameters
    ----------
    probability : float
        Predicted probability of UTI (0-1)

    Returns
    -------
    tuple
        (severity_category, severity_multiplier)
    """
    if probability < 0.4:
        return 'mild', ODE_PARAMETERS['severity_multiplier_mild']['value']
    elif probability < 0.7:
        return 'moderate', ODE_PARAMETERS['severity_multiplier_moderate']['value']
    else:
        return 'severe', ODE_PARAMETERS['severity_multiplier_severe']['value']


def sad_model(y: np.ndarray, t: float, beta: float, gamma: float,
              delta: float, mu: float) -> List[float]:
    """
    SAD (Susceptible-Affected-Diseased) ODE model for UTI progression.

    Parameters
    ----------
    y : array
        Current state [S, A, D]
    t : float
        Time point
    beta : float
        Infection/transmission rate
    gamma : float
        Progression rate (A -> D)
    delta : float
        Recovery rate (D -> S)
    mu : float
        Natural resolution rate (A -> S)

    Returns
    -------
    list
        Derivatives [dS/dt, dA/dt, dD/dt]
    """
    S, A, D = y
    N = S + A + D  # Total population

    # Ensure non-negative compartments
    S = max(0, S)
    A = max(0, A)
    D = max(0, D)

    # ODE equations
    dSdt = -beta * S * A / N + delta * D + mu * A
    dAdt = beta * S * A / N - gamma * A - mu * A
    dDdt = gamma * A - delta * D

    return [dSdt, dAdt, dDdt]


def simulate_progression(
    initial_conditions: Dict[str, float],
    severity_level: str,
    t_max: int = 90,
    dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate UTI disease progression for a given severity level.

    Parameters
    ----------
    initial_conditions : dict
        Initial values for S, A, D compartments
    severity_level : str
        'mild', 'moderate', or 'severe'
    t_max : int
        Maximum simulation time (days)
    dt : float
        Time step (days)

    Returns
    -------
    tuple
        (time_points, solution_array)
    """
    # Get base parameters
    beta_base = ODE_PARAMETERS['beta_base']['value']
    gamma = ODE_PARAMETERS['gamma']['value']
    delta = ODE_PARAMETERS['delta']['value']
    mu = ODE_PARAMETERS['mu']['value']

    # Apply severity multiplier to beta
    severity_key = f'severity_multiplier_{severity_level}'
    severity_mult = ODE_PARAMETERS[severity_key]['value']
    beta = beta_base * severity_mult

    # Initial state
    y0 = [
        initial_conditions.get('S', 0.9),
        initial_conditions.get('A', 0.1),
        initial_conditions.get('D', 0.0)
    ]

    # Time points
    t = np.arange(0, t_max, dt)

    # Solve ODE
    solution = odeint(sad_model, y0, t, args=(beta, gamma, delta, mu))

    return t, solution


def run_scenario_analysis() -> Dict:
    """
    Run disease progression simulations for all severity scenarios.

    Returns
    -------
    dict
        Simulation results for each scenario
    """
    # Standard initial conditions
    # Assume 90% susceptible, 10% affected (early infection), 0% diseased
    initial_conditions = {'S': 0.90, 'A': 0.10, 'D': 0.00}

    scenarios = ['mild', 'moderate', 'severe']
    results = {}

    for scenario in scenarios:
        t, solution = simulate_progression(initial_conditions, scenario, t_max=90)
        results[scenario] = {
            'time': t,
            'S': solution[:, 0],
            'A': solution[:, 1],
            'D': solution[:, 2],
            'parameters': {
                'beta': ODE_PARAMETERS['beta_base']['value'] * \
                        ODE_PARAMETERS[f'severity_multiplier_{scenario}']['value'],
                'gamma': ODE_PARAMETERS['gamma']['value'],
                'delta': ODE_PARAMETERS['delta']['value'],
                'mu': ODE_PARAMETERS['mu']['value']
            }
        }

        # Calculate key metrics
        peak_diseased_idx = np.argmax(solution[:, 2])
        results[scenario]['metrics'] = {
            'peak_diseased': float(solution[peak_diseased_idx, 2]),
            'time_to_peak': float(t[peak_diseased_idx]),
            'final_susceptible': float(solution[-1, 0]),
            'total_infected': float(1 - solution[-1, 0])
        }

    return results


def sensitivity_analysis(param_name: str, param_range: np.ndarray,
                         severity: str = 'moderate') -> Dict:
    """
    Perform sensitivity analysis for a single parameter.

    Parameters
    ----------
    param_name : str
        Parameter to vary ('beta', 'gamma', 'delta', 'mu')
    param_range : np.ndarray
        Range of parameter values to test
    severity : str
        Severity level for baseline parameters

    Returns
    -------
    dict
        Sensitivity analysis results
    """
    initial_conditions = {'S': 0.90, 'A': 0.10, 'D': 0.00}
    t_max = 90

    # Get baseline parameters
    severity_mult = ODE_PARAMETERS[f'severity_multiplier_{severity}']['value']
    base_params = {
        'beta': ODE_PARAMETERS['beta_base']['value'] * severity_mult,
        'gamma': ODE_PARAMETERS['gamma']['value'],
        'delta': ODE_PARAMETERS['delta']['value'],
        'mu': ODE_PARAMETERS['mu']['value']
    }

    results = {
        'param_name': param_name,
        'param_values': param_range.tolist(),
        'peak_diseased': [],
        'time_to_peak': [],
        'final_susceptible': []
    }

    for value in param_range:
        # Update parameter
        params = base_params.copy()
        params[param_name] = value

        # Initial state
        y0 = [initial_conditions['S'], initial_conditions['A'], initial_conditions['D']]
        t = np.arange(0, t_max, 0.1)

        # Solve
        solution = odeint(sad_model, y0, t,
                         args=(params['beta'], params['gamma'],
                               params['delta'], params['mu']))

        # Extract metrics
        peak_idx = np.argmax(solution[:, 2])
        results['peak_diseased'].append(float(solution[peak_idx, 2]))
        results['time_to_peak'].append(float(t[peak_idx]))
        results['final_susceptible'].append(float(solution[-1, 0]))

    return results


def plot_disease_progression(results: Dict, save_path: Path) -> None:
    """
    Plot disease progression curves for all scenarios.

    Parameters
    ----------
    results : dict
        Simulation results from run_scenario_analysis()
    save_path : Path
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'S': '#2ecc71', 'A': '#f1c40f', 'D': '#e74c3c'}
    linestyles = {'mild': '-', 'moderate': '--', 'severe': '-.'}

    for scenario, data in results.items():
        t = data['time']

        for i, (compartment, color) in enumerate(colors.items()):
            axes[i].plot(t, data[compartment], label=f'{scenario.capitalize()}',
                        linestyle=linestyles[scenario], color=color if scenario == 'moderate' else None,
                        lw=2)

    compartment_names = ['Susceptible (S)', 'Affected (A)', 'Diseased (D)']

    for i, (ax, name) in enumerate(zip(axes, compartment_names)):
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Proportion of Population', fontsize=12)
        ax.set_title(name, fontsize=14)
        ax.legend(loc='best')
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('SAD Model: UTI Disease Progression by Severity Level', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_progression(results: Dict, save_path: Path) -> None:
    """
    Plot combined disease progression showing all compartments.

    Parameters
    ----------
    results : dict
        Simulation results
    save_path : Path
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'S': '#2ecc71', 'A': '#f1c40f', 'D': '#e74c3c'}
    scenarios = ['mild', 'moderate', 'severe']
    titles = ['Mild Severity (P < 0.4)', 'Moderate Severity (0.4 <= P < 0.7)',
              'Severe (P >= 0.7)']

    for i, (scenario, title) in enumerate(zip(scenarios, titles)):
        t = results[scenario]['time']

        # Plot stacked area
        axes[i].fill_between(t, 0, results[scenario]['D'],
                             color=colors['D'], alpha=0.7, label='Diseased (D)')
        axes[i].fill_between(t, results[scenario]['D'],
                             results[scenario]['D'] + results[scenario]['A'],
                             color=colors['A'], alpha=0.7, label='Affected (A)')
        axes[i].fill_between(t, results[scenario]['D'] + results[scenario]['A'],
                             results[scenario]['D'] + results[scenario]['A'] + results[scenario]['S'],
                             color=colors['S'], alpha=0.7, label='Susceptible (S)')

        axes[i].set_xlabel('Time (days)', fontsize=12)
        axes[i].set_ylabel('Proportion', fontsize=12)
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlim(0, 90)
        axes[i].set_ylim(0, 1)
        axes[i].legend(loc='upper right')

        # Add metrics annotation
        metrics = results[scenario]['metrics']
        metrics_text = f"Peak D: {metrics['peak_diseased']:.2%}\n" \
                      f"Time to peak: {metrics['time_to_peak']:.1f} days"
        axes[i].text(0.95, 0.05, metrics_text, transform=axes[i].transAxes,
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('SAD Model: Compartment Dynamics by Severity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sensitivity_heatmap(save_path: Path) -> None:
    """
    Generate sensitivity analysis heatmap for key parameters.

    Parameters
    ----------
    save_path : Path
        Path to save figure
    """
    # Parameters to analyze
    params_to_test = {
        'beta': np.linspace(0.05, 0.30, 10),
        'gamma': np.linspace(0.10, 0.40, 10),
        'delta': np.linspace(0.05, 0.25, 10),
        'mu': np.linspace(0.02, 0.15, 10)
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    param_descriptions = {
        'beta': 'Infection Rate (β)',
        'gamma': 'Progression Rate (γ)',
        'delta': 'Recovery Rate (δ)',
        'mu': 'Natural Resolution (μ)'
    }

    for i, (param, values) in enumerate(params_to_test.items()):
        results = sensitivity_analysis(param, values)

        # Plot peak diseased vs parameter value
        axes[i].plot(values, results['peak_diseased'], 'o-', color='#e74c3c',
                    lw=2, markersize=6, label='Peak Diseased')
        axes[i].set_xlabel(param_descriptions[param], fontsize=11)
        axes[i].set_ylabel('Peak Diseased Proportion', fontsize=11)
        axes[i].set_title(f'Sensitivity to {param_descriptions[param]}', fontsize=12)
        axes[i].grid(True, alpha=0.3)

        # Add baseline marker
        baseline = ODE_PARAMETERS[param if param != 'beta' else 'beta_base']['value']
        if param == 'beta':
            baseline *= ODE_PARAMETERS['severity_multiplier_moderate']['value']
        axes[i].axvline(x=baseline, color='gray', linestyle='--', alpha=0.7,
                       label='Baseline')
        axes[i].legend()

    plt.suptitle('Parameter Sensitivity Analysis - SAD ODE Model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_parameter_table() -> pd.DataFrame:
    """
    Generate ODE parameter table for manuscript.

    Returns
    -------
    pd.DataFrame
        Parameter table
    """
    rows = []

    for param_name, param_info in ODE_PARAMETERS.items():
        rows.append({
            'Parameter': param_name,
            'Value': param_info['value'],
            'Unit': param_info['unit'],
            'Description': param_info['description'],
            'Source': param_info['source']
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'ode_parameters.csv', index=False)
    print(f"Saved: {TABLES_DIR / 'ode_parameters.csv'}")

    return df


def generate_simulation_outcomes_table(results: Dict) -> pd.DataFrame:
    """
    Generate simulation outcomes table for manuscript.

    Parameters
    ----------
    results : dict
        Simulation results

    Returns
    -------
    pd.DataFrame
        Outcomes table
    """
    rows = []

    for scenario in ['mild', 'moderate', 'severe']:
        metrics = results[scenario]['metrics']
        params = results[scenario]['parameters']

        rows.append({
            'Severity': scenario.capitalize(),
            'Effective Beta': f"{params['beta']:.3f}",
            'Peak Diseased (%)': f"{metrics['peak_diseased']*100:.1f}",
            'Time to Peak (days)': f"{metrics['time_to_peak']:.1f}",
            'Final Susceptible (%)': f"{metrics['final_susceptible']*100:.1f}",
            'Total Infected (%)': f"{metrics['total_infected']*100:.1f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'ode_simulation_outcomes.csv', index=False)
    print(f"Saved: {TABLES_DIR / 'ode_simulation_outcomes.csv'}")

    return df


def apply_model_to_predictions(model, X: np.ndarray) -> pd.DataFrame:
    """
    Apply SAD ODE model to ML predictions for individual patients.

    Parameters
    ----------
    model : XGBClassifier
        Trained classification model
    X : np.ndarray
        Feature matrix

    Returns
    -------
    pd.DataFrame
        Patient-level predictions with severity and progression metrics
    """
    # Get predictions
    probabilities = model.predict_proba(X)[:, 1]

    results = []

    for i, prob in enumerate(probabilities):
        severity, mult = get_severity_category(prob)

        # Run individual simulation
        initial_conditions = {'S': 0.9, 'A': 0.1, 'D': 0.0}
        t, solution = simulate_progression(initial_conditions, severity, t_max=60)

        # Extract metrics
        peak_idx = np.argmax(solution[:, 2])

        results.append({
            'patient_id': i + 1,
            'uti_probability': prob,
            'severity_category': severity,
            'severity_multiplier': mult,
            'peak_diseased': solution[peak_idx, 2],
            'time_to_peak_days': t[peak_idx],
            'day30_diseased': solution[np.argmin(np.abs(t - 30)), 2],
            'day60_susceptible': solution[-1, 0]
        })

    return pd.DataFrame(results)


def main():
    """Main SAD ODE modeling pipeline."""
    print("=" * 60)
    print("UTI PREDICTION - SAD ODE DISEASE PROGRESSION MODEL")
    print("=" * 60)

    # Step 1: Generate parameter table
    print("\n[Step 1] Generating ODE parameter table...")
    param_table = generate_parameter_table()
    print("\nODE Parameters:")
    print(param_table[['Parameter', 'Value', 'Unit']].to_string(index=False))

    # Note about citations
    print("\n" + "-" * 60)
    print("NOTE: Parameter sources marked [CITATION NEEDED] require")
    print("      literature verification before manuscript submission.")
    print("      See References/citations_log.md")
    print("-" * 60)

    # Step 2: Run scenario analysis
    print("\n[Step 2] Running scenario simulations...")
    simulation_results = run_scenario_analysis()

    # Step 3: Generate outcomes table
    print("\n[Step 3] Generating simulation outcomes table...")
    outcomes_table = generate_simulation_outcomes_table(simulation_results)
    print("\nSimulation Outcomes:")
    print(outcomes_table.to_string(index=False))

    # Step 4: Generate visualizations
    print("\n[Step 4] Generating visualizations...")
    plot_disease_progression(simulation_results,
                            FIGURES_DIR / 'fig18_ode_progression_curves.png')
    plot_combined_progression(simulation_results,
                             FIGURES_DIR / 'fig19_ode_combined_dynamics.png')
    plot_sensitivity_heatmap(FIGURES_DIR / 'fig20_ode_sensitivity_analysis.png')

    # Step 5: Apply to ML predictions
    print("\n[Step 5] Applying ODE model to ML predictions...")

    # Load model and data
    try:
        model = joblib.load(MODELS_DIR / 'xgboost_model.joblib')
        data = np.load(RESULTS_DIR / 'train_test_split.npz')
        X_test = data['X_test']
        y_test = data['y_test']

        patient_predictions = apply_model_to_predictions(model, X_test)
        patient_predictions['true_label'] = y_test

        patient_predictions.to_csv(RESULTS_DIR / 'patient_ode_predictions.csv', index=False)
        print(f"Saved: {RESULTS_DIR / 'patient_ode_predictions.csv'}")

        # Summary by severity
        severity_summary = patient_predictions.groupby('severity_category').agg({
            'uti_probability': 'mean',
            'peak_diseased': 'mean',
            'time_to_peak_days': 'mean',
            'patient_id': 'count'
        }).rename(columns={'patient_id': 'n_patients'})

        print("\nPatient Severity Distribution:")
        print(severity_summary.to_string())

    except FileNotFoundError:
        print("Model not found - skipping patient-level predictions")

    # Step 6: Save complete results
    print("\n[Step 6] Saving complete results...")

    results = {
        'model_type': 'SAD (Susceptible-Affected-Diseased)',
        'parameters': {k: v['value'] for k, v in ODE_PARAMETERS.items()},
        'scenarios': {
            scenario: {
                'metrics': data['metrics'],
                'parameters': data['parameters']
            }
            for scenario, data in simulation_results.items()
        }
    }

    with open(RESULTS_DIR / 'ode_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'ode_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SAD ODE MODELING COMPLETE")
    print("=" * 60)
    print(f"\nModel: Susceptible-Affected-Diseased (SAD)")
    print(f"\nKey Findings by Severity:")

    for scenario in ['mild', 'moderate', 'severe']:
        metrics = simulation_results[scenario]['metrics']
        print(f"\n  {scenario.upper()}:")
        print(f"    - Peak diseased: {metrics['peak_diseased']:.1%}")
        print(f"    - Time to peak: {metrics['time_to_peak']:.1f} days")
        print(f"    - Total infected: {metrics['total_infected']:.1%}")

    print(f"\nOutputs saved to:")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Tables: {TABLES_DIR}")
    print(f"  - Results: {RESULTS_DIR}")

    return simulation_results, param_table


if __name__ == "__main__":
    simulation_results, param_table = main()
