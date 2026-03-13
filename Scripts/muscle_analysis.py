# muscle_analysis.py
# This script performs initial exploration of the Muscle Strength dataset.

import pandas as pd

print("Libraries loaded")

# Load dataset
muscle = pd.read_csv("Original-Data/F_PROACT_MUSCLESTRENGTH.csv")

print("\nDataset loaded successfully")

# Dataset shape
print("\nDataset shape:")
print(muscle.shape)

# Column names
print("\nColumns in dataset:")
print(muscle.columns)

# First rows
print("\nFirst 5 rows of the dataset:")
print(muscle.head())

# Missing values
print("\nMissing values per column:")
print(muscle.isnull().sum())

# Unique patients
print("\nNumber of unique patients:")
print(muscle["subject_id"].nunique())