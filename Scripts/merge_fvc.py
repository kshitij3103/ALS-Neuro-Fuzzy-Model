# merge_fvc.py
# This script merges the cleaned FVC dataset with the existing
# ALSFRS + Demographics dataset.

import pandas as pd

print("Libraries loaded")

# --------------------------------------------------
# Step 1: Load datasets
# --------------------------------------------------

als_demo = pd.read_csv("PreProcessed-Data/alsfrs_demographics.csv")
fvc = pd.read_csv("PreProcessed-Data/fvc_cleaned.csv")

print("\nALS + Demographics shape:", als_demo.shape)
print("FVC dataset shape:", fvc.shape)


# --------------------------------------------------
# Step 2: Merge datasets using subject_id
# --------------------------------------------------

df = pd.merge(
    als_demo,
    fvc,
    on="subject_id",
    how="inner"
)

print("\nMerged dataset shape:", df.shape)


# --------------------------------------------------
# Step 3: Save merged dataset
# --------------------------------------------------

df.to_csv("PreProcessed-Data/alsfrs_demo_fvc.csv", index=False)

print("\nMerged dataset saved successfully")