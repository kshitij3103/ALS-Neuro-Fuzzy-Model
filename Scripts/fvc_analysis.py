# fvc_cleaning.py
# This script cleans the FVC dataset and creates a robust FVC feature
# by combining multiple trial measurements.

import pandas as pd

print("Libraries loaded")


# ----------------------------------------------------
# Step 1: Load the raw FVC dataset
# ----------------------------------------------------
fvc = pd.read_csv("Original-Data/F_PROACT_FVC.csv")

print("\nDataset loaded")
print("Shape:", fvc.shape)


# ----------------------------------------------------
# Step 2: Select relevant columns
# These columns contain the three FVC trial measurements
# ----------------------------------------------------
fvc = fvc[[
    "subject_id",
    "Subject_Liters_Trial_1",
    "Subject_Liters_Trial_2",
    "Subject_Liters_Trial_3"
]]

print("\nSelected columns:")
print(fvc.head())


# ----------------------------------------------------
# Step 3: Create a combined FVC feature
# Compute the mean across available trials
# Missing values are ignored during the mean calculation
# ----------------------------------------------------
fvc["FVC_mean"] = fvc[[
    "Subject_Liters_Trial_1",
    "Subject_Liters_Trial_2",
    "Subject_Liters_Trial_3"
]].mean(axis=1, skipna=True)

print("\nNew FVC feature created")


# ----------------------------------------------------
# Step 4: Handle remaining missing values
# Fill missing values with the median FVC
# ----------------------------------------------------
fvc["FVC_mean"] = fvc["FVC_mean"].fillna(fvc["FVC_mean"].median())

print("\nMissing values after filling:")
print(fvc.isnull().sum())


# ----------------------------------------------------
# Step 5: Keep only required columns
# ----------------------------------------------------
fvc = fvc[[
    "subject_id",
    "FVC_mean"
]]


# ----------------------------------------------------
# Step 6: Keep one record per patient
# Since patients have multiple visits, we keep the latest entry
# ----------------------------------------------------
fvc = fvc.groupby("subject_id").last().reset_index()

print("\nDataset after grouping by patient:")
print(fvc.shape)


# ----------------------------------------------------
# Step 7: Save the cleaned dataset
# ----------------------------------------------------
fvc.to_csv("PreProcessed-Data/fvc_cleaned.csv", index=False)

print("\nCleaned FVC dataset saved successfully")