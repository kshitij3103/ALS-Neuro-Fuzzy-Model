# 🧠 ALS Disease Progression Prediction using Neuro-Fuzzy Systems

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/AI-Neuro--Fuzzy-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **An Adaptive Neuro-Fuzzy Inference System (ANFIS) designed to model and forecast the progression of Amyotrophic Lateral Sclerosis (ALS) using longitudinal clinical trial data.**

## 📑 Table of Contents
- [The Problem: Deep Learning's "Black Box"](#the-problem-deep-learnings-black-box)
- [The Solution: Why Neuro-Fuzzy?](#the-solution-why-neuro-fuzzy)
- [Project Content: How the Model Works](#project-content-how-the-model-works)
- [Dataset & Important Notice](#dataset--important-notice)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)

---

## 🛑 The Problem: Deep Learning's "Black Box"
While traditional Deep Learning (DL) models are powerful at finding hidden patterns in clinical data, they suffer from a major critical flaw in healthcare applications: **a complete lack of interpretability.** When a standard Neural Network predicts a rapid decline in an ALS patient, it cannot explain *why* it reached that conclusion. It acts as a "black box," making it incredibly difficult for clinicians to trust the output, validate the reasoning, or use the model to make life-altering patient care decisions. Furthermore, DL models often struggle to handle the high degree of uncertainty, noise, and missing values inherent in longitudinal medical records.

## 💡 The Solution: Why Neuro-Fuzzy?
This project overcomes the limitations of deep learning by implementing an **Adaptive Neuro-Fuzzy Inference System (ANFIS)**. 

ANFIS bridges the gap between machine learning and human reasoning. It extracts the raw predictive power of artificial neural networks and maps it onto the mathematical framework of Fuzzy Logic. 

**Key Improvements over Standard Deep Learning:**
1. **Total Interpretability:** Instead of hidden weights and biases, the ANFIS model generates explicit, human-readable **IF-THEN rules** (e.g., *IF Creatinine is LOW and CK is HIGH, THEN Progression is RAPID*).
2. **Handling Clinical Uncertainty:** Medical data isn't strictly binary. Fuzzy logic allows the model to process "degrees of truth" (e.g., a lab value being 70% "Normal" and 30% "Elevated"), perfectly mirroring how doctors actually interpret borderline lab results.
3. **Clinician Trust:** By outputting both an accurate mathematical prediction and the logical rules used to get there, this model serves as a transparent decision-support tool rather than an opaque oracle.

---

## ⚙️ Project Content: How the Model Works
This repository contains the complete pipeline for ingesting patient laboratory data and training the ANFIS architecture. The core workflow of the project does the following:

1. **Data Ingestion & Preprocessing:** Cleans longitudinal lab data from the PRO-ACT database, handling the specific noise and irregular time-intervals found in ALS clinical trials.
2. **Fuzzification:** The model takes raw numerical lab values (e.g., Blood Urea Nitrogen, Creatinine) and converts them into "fuzzy sets" with specific membership functions (Low, Normal, High).
3. **Neural Training:** A neural network architecture (typically using backpropagation and least squares estimation) learns the optimal shape of these fuzzy sets and the best rules to connect them based on historical patient outcomes.
4. **Defuzzification & Forecasting:** The model evaluates a new patient's current lab markers against the learned rules to output a crisp, numerical prediction of their future ALSFRS-R score (disease progression rate).

---

## 📊 Dataset & Important Notice
This project relies on the **PRO-ACT (Pooled Resource Open-Access ALS Clinical Trials)** database.

🚨 **WARNING: Large Dataset Excluded** 🚨
To comply with GitHub's file size limits, the primary raw data file (`F_PROACT_LABS.csv` - ~138 MB) is **not** included in this repository. 

**To run this project locally:**
1. Download the `F_PROACT_LABS.csv` file from the PRO-ACT database.
2. Place the file precisely at: `data/raw-data/F_PROACT_LABS.csv`

---

## 🏗️ Project Architecture
```text
ALS-Neuro-Fuzzy-Model/
│
├── data/
│   ├── raw-data/         # Place F_PROACT_LABS.csv here
│   └── processed-data/   # Cleaned, normalized datasets
│
├── src/                  # Source code for data preprocessing and ANFIS implementation
├── notebooks/            # Jupyter notebooks for exploratory data analysis
├── .gitignore
└── README.md
