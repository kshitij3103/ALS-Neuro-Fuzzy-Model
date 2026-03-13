# vitals_analysis.py
# This script performs initial exploration of the Vital Signs dataset.

import pandas as pd

print("Libraries loaded")

# Load dataset
vitals = pd.read_csv("Original-Data/F_PROACT_VITALSIGNS.csv")

print("\nDataset loaded successfully")

# Dataset shape
print("\nDataset shape:")
print(vitals.shape)

# Column names
print("\nColumns in dataset:")
print(vitals.columns)

# First rows
print("\nFirst 5 rows:")
print(vitals.head())

# Missing values
print("\nMissing values per column:")
print(vitals.isnull().sum())

# Unique patients
print("\nUnique patients:")
print(vitals["subject_id"].nunique())