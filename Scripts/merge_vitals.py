# merge_vitals.py
# Merge Vital Signs with existing ALS dataset

import pandas as pd

print("Libraries loaded")

# Load datasets
main_df = pd.read_csv("PreProcessed-Data/alsfrs_demo_fvc.csv")
vitals = pd.read_csv("PreProcessed-Data/vitals_cleaned.csv")

print("\nMain dataset shape:", main_df.shape)
print("Vitals dataset shape:", vitals.shape)

# Merge using subject_id
merged = pd.merge(main_df, vitals, on="subject_id", how="inner")

print("\nMerged dataset shape:")
print(merged.shape)

# Save merged dataset
merged.to_csv("PreProcessed-Data/alsfrs_demo_fvc_vitals.csv", index=False)

print("\nMerged dataset saved successfully")