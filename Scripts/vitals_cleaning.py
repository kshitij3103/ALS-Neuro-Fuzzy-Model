# vitals_cleaning.py
# Clean and prepare Vital Signs dataset

import pandas as pd

print("Libraries loaded")

# Load dataset
vitals = pd.read_csv("Original-Data/F_PROACT_VITALSIGNS.csv")

print("\nDataset loaded")
print("Shape:", vitals.shape)

# Select useful columns
vitals = vitals[[
    "subject_id",
    "Blood_Pressure_Diastolic",
    "Blood_Pressure_Systolic",
    "Pulse",
    "Respiratory_Rate",
    "Weight"
]]

print("\nSelected columns:")
print(vitals.head())

# Convert columns to numeric
for col in vitals.columns:
    if col != "subject_id":
        vitals[col] = pd.to_numeric(vitals[col], errors="coerce")

# Fill missing values with median
for col in vitals.columns:
    if col != "subject_id":
        vitals[col] = vitals[col].fillna(vitals[col].median())

print("\nMissing values after filling:")
print(vitals.isnull().sum())

# Keep latest record per patient
vitals = vitals.groupby("subject_id").last().reset_index()

print("\nDataset after grouping by patient:")
print(vitals.shape)

# Save cleaned dataset
vitals.to_csv("PreProcessed-Data/vitals_cleaned.csv", index=False)

print("\nCleaned Vital Signs dataset saved")