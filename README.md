# UTI Prediction with SHAP Explainability and SAD-ODE Progression Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official code implementation** for the paper: *"Transformer-Based and Explainable Learning for Urinary Tract Infection Prediction with SAD-ODE Progression Analysis"*

## Overview

This repository provides a complete machine learning pipeline for UTI prediction featuring:

- **XGBoost Classifier** with Bayesian hyperparameter optimization (Optuna)
- **FT-Transformer** (Feature Tokenizer Transformer) for tabular deep learning
- **SHAP Explainability** with TreeSHAP for model interpretability
- **SAD-ODE Model** (Susceptible-Affected-Diseased) for disease progression simulation
- **4-Class Severity Mapping** with per-patient probability outputs

## Repository Structure

```
├── 01_preprocessing.py              # Data preprocessing and EDA
├── 02_xgboost_baseline.py           # XGBoost with Optuna tuning
├── 03_ft_transformer.py             # FT-Transformer implementation
├── 04_explainability.py             # SHAP analysis and visualizations
├── 05_sad_ode_model.py              # SAD-ODE disease progression model
├── 06_results_generator.py          # Results consolidation
├── 07_patient_probability_output.py # Per-patient severity probabilities
├── utils/
│   ├── data_utils.py                # Data loading utilities
│   ├── model_utils.py               # Model helper functions
│   ├── plot_utils.py                # Visualization utilities
│   ├── latex_utils.py               # LaTeX table generation
│   └── citation_utils.py            # Citation verification
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/khurrameycon/uti-prediction-shap-ode.git
cd uti-prediction-shap-ode

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Execute scripts sequentially:

```bash
# Step 1: Data Preprocessing
python 01_preprocessing.py

# Step 2: XGBoost Model Training
python 02_xgboost_baseline.py

# Step 3: FT-Transformer Training
python 03_ft_transformer.py

# Step 4: SHAP Explainability Analysis
python 04_explainability.py

# Step 5: SAD-ODE Disease Progression
python 05_sad_ode_model.py

# Step 6: Generate Results
python 06_results_generator.py

# Step 7: Per-Patient Probability Outputs
python 07_patient_probability_output.py
```

## Model Performance

| Model | Accuracy | AUC-ROC | F1 Score | MCC |
|-------|----------|---------|----------|-----|
| XGBoost | 100% | 1.000 | 1.000 | 1.000 |
| FT-Transformer | 75% | 1.000 | 0.857 | 0.000 |

## Top Predictive Features (SHAP)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 1 | Urinary Urgency | 1.608 |
| 2 | Symptom Count | 1.338 |
| 3 | Painful Urination | 0.134 |
| 4 | Body Temperature | 0.064 |
| 5 | Lumbar Pain | 0.016 |

## SAD-ODE Model

Compartmental disease progression model:

```
dS/dt = -β(severity) × S × A
dA/dt = β × S × A - γ × A
dD/dt = γ × A
```

| Severity | Peak Diseased | Time to Peak | Total Infected |
|----------|---------------|--------------|----------------|
| Mild | 4.6% | 6.3 days | 0.0% |
| Moderate | 5.8% | 7.8 days | 0.0% |
| Severe | 12.5% | 18.7 days | 16.7% |

## Per-Patient Probability Output

Generates Excel files with 4-class severity probabilities:

```
Patient_ID | Prob_No_Impairment | Prob_Very_Mild | Prob_Mild | Prob_Moderate
-----------|--------------------|----------------|-----------|---------------
patient_001| 0.45               | 0.38           | 0.14      | 0.02
patient_002| 0.00               | 0.06           | 0.29      | 0.65
```

## Dataset

UTI dataset (n=120) with 8 features:
- 6 clinical symptoms (binary)
- 1 temperature measurement (continuous)
- 1 engineered feature (symptom count)

## Citation

```bibtex
@article{khurram2024uti,
  title={Transformer-Based and Explainable Learning for Urinary Tract
         Infection Prediction with SAD-ODE Progression Analysis},
  author={Khurram, SL and et al.},
  journal={Scientific Reports},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Dataset: UCI Machine Learning Repository / Kaggle
- SHAP Library for model interpretability
- PyTorch for deep learning framework
