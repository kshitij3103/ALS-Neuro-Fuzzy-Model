# ALS Progression Prediction: An Explainable Cluster-Based ANFIS Approach

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Authors:** Kshitij Kataria & Yash Chaudhary

This repository contains the official codebase for predicting the functional progression of Amyotrophic Lateral Sclerosis (ALS) using a hybrid-learning **Adaptive Neuro-Fuzzy Inference System (ANFIS)**. 

The primary contribution of this work is an interpretable, parameter-efficient prognostic tool that achieves highly competitive performance against black-box Deep Learning (DL) models while utilizing significantly fewer parameters (~160 vs >3000) and exclusively relying on 5 clinical features.

---

## Table of Contents
- [Background & Motivation](#background--motivation)
- [Key Features](#key-features)
- [Project Workflow (How It Works)](#project-workflow-how-it-works)
- [Software & Dependencies](#software--dependencies)
- [Repository Structure](#repository-structure)
- [Data & Reproducibility](#data--reproducibility)
- [How to Run](#how-to-run)
- [Model Methodology](#model-methodology)

---

## Background & Motivation
ALS is a devastating neurodegenerative disease with highly heterogeneous disease progression. While recent Deep Learning models have demonstrated high predictive accuracy on datasets like PRO-ACT, they inherently act as "black boxes." In critical clinical settings, physicians require **transparency and interpretability** to trust a model's prognosis.

Our ANFIS approach bridges this gap by generating explicit, linguistic IF-THEN fuzzy rules (e.g., *IF FVC is low AND Age is high THEN Progression is Fast*). This framework provides a clear interpretability-versus-capacity trade-off, offering actionable insights for clinical trial stratification and patient counseling.

---

## Key Features
- **Intrinsic Interpretability:** Extracts a 10-rule fuzzy rule base that directly mirrors clinical reasoning, completely bypassing the need for post-hoc explainers like SHAP/LIME.
- **Parameter Efficiency:** The entire ensemble operates on roughly ~160 parameters, demonstrating robust resistance to overfitting and a very low data-acquisition burden.
- **Strict Data Hygiene:** Implementation of rigorous patient-level `KFold` cross-validation with internal, dynamic IQR-based outlier mitigation and median imputation to ensure zero data leakage between folds.
- **Automated Pipeline:** 1-click execution for the entire workflow—from raw clinical tables to final visual analytics and rule-extraction tables.

---

## Project Workflow (How It Works)
The pipeline is divided into 4 core phases, each handled by a dedicated script:

1. **Phase 1: Preprocessing (`step1_preprocessing.py`)** 
   - Cleans the raw PRO-ACT clinical tables (Vitals, ALSFRS, FVC).
   - Computes the true functional decline slope (ALSFRS-R points lost per month) between the 4th and 12th month for each patient. This serves as our target variable.

2. **Phase 2: Feature Engineering (`step2_feature_engineering.py`)** 
   - Extracts clinical data exclusively from the first 3 months of observation (to prevent data leakage).
   - Generates summary statistics (min, max, median, slope) for longitudinal measurements. 
   - Locks in the final 5 clinical features (e.g., Onset Duration, FVC slope, Speech score).

3. **Phase 3: Model Training (`step3_model_training.py`)** 
   - Trains the baseline Machine Learning models (Random Forest, XGBoost) alongside the PyTorch-based **ANFIS ensemble**.
   - Uses strict 5-Fold Patient-Level Cross Validation with internal IQR clipping to ensure the models don't overfit to outliers.

4. **Phase 4: Visualization (`step4_visualization.py`)** 
   - Generates the final predicted-vs-true scatter plots and error distribution graphs.
   - Explicitly extracts the 10 trained IF-THEN fuzzy rules from the ANFIS model into human-readable clinical tables.

---

## Software & Dependencies
The project is built on **Python 3.12+**. Core dependencies include:
- `torch` (ANFIS neural architecture & optimization)
- `scikit-learn` (Cross-validation, pre-processing, and baselines)
- `xgboost` (Gradient boosted baselines)
- `pandas` & `numpy` (Data manipulation)
- `matplotlib` & `seaborn` (High-quality, print-ready visualizations)
- `scipy` (Statistical testing and Pearson Correlation measurements)

**Installation:**
```bash
pip install -r requirements.txt
# Alternatively:
pip install torch scikit-learn pandas numpy matplotlib seaborn xgboost scipy
```

---

## Repository Structure
```text
.
├── main.py                     # Master execution script (Runs the entire pipeline end-to-end)
├── step1_preprocessing.py      # Phase 1: Cleans raw clinical tables & computes target slopes
├── step2_feature_engineering.py # Phase 2: Extracts longitudinal metrics (Min, Max, Median, Slope)
├── step3_model_training.py     # Phase 3: Trains baselines and the cross-validated ANFIS ensemble
├── step4_visualization.py      # Phase 4: Generates print-ready plots, rules tables, and results
├── data/
│   ├── raw-data/               # Original clinical files (CSV format)
│   └── preprocessed-data/      # Master dataset and trained PyTorch model weights (.pth)
├── analysis/                   # Output directory for plots, correlation matrices, and metrics
└── scripts/
    └── detailed scripts/       # Full library of 30+ granular, modular development scripts
```

---

## Data & Reproducibility 
To comply with open-science and peer-review standards for clinical ML research:
- **Dataset:** This study utilizes the PRO-ACT (Pooled Resource Open-Access ALS Clinical Trials) database. Due to strict data privacy regulations, the raw PRO-ACT database is only available to registered researchers via the [PRO-ACT portal](https://nctu.partners.org/ProACT).
- **Reproducibility:** All data-processing scripts, inclusion filters, and model code used in this study are provided in this repository. To guarantee exact reproducibility, a global random seed (`np.random.seed(42)`, `torch.manual_seed(42)`) is enforced across all data splits, KMeans initializations, and model training loops.

---

## How to Run

> **Execution Time:** The complete pipeline takes approximately **10-15 minutes** depending on your local hardware.

**Option 1: End-to-End Execution (Recommended)**  
Run the entire project from raw data parsing to final evaluation with a single command:
```bash
python main.py
```
*Note: The `analysis/` folder will be automatically refreshed and populated with new, high-resolution visual outputs upon completion.*

**Option 2: Modular Execution**  
If you need to isolate specific phases of the workflow, execute the numbered steps sequentially:
```bash
python step1_preprocessing.py
python step2_feature_engineering.py
python step3_model_training.py
python step4_visualization.py
```

---

## Model Methodology: Leakage-Free ANFIS
- **Target Variable:** The continuous ALSFRS-R functional decline slope between the 3-month ($t_1$) and 12-month ($t_2$) clinical observation window.
- **Input Features:** Extracted exclusively from the $t_0$ to 3-month window. Restricted to 5 strictly selected features to prevent dimensionality explosion.
- **Hybrid Learning:** Employs **K-Means clustering** for premise initialization, **Gradient Descent (Adam)** for Gaussian width ($\sigma$) optimization, and **Recursive Least Squares Estimation (LSE)** for consequent parameter tuning.
