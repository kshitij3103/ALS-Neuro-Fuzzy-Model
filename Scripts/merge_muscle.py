# merge_muscle.py
# Merge Muscle Strength dataset with the main ALS dataset

import pandas as pd

print("Libraries loaded")

# Load datasets
main_df = pd.read_csv("PreProcessed-Data/alsfrs_demo_fvc_vitals.csv")
muscle = pd.read_csv("PreProcessed-Data/muscle_cleaned.csv")

print("\nMain dataset shape:", main_df.shape)
print("Muscle dataset shape:", muscle.shape)

# Merge datasets
merged = pd.merge(main_df, muscle, on="subject_id", how="inner")

print("\nMerged dataset shape:")
print(merged.shape)

# Save merged dataset
merged.to_csv("PreProcessed-Data/alsfrs_demo_fvc_vitals_muscle.csv", index=False)

print("\nMerged dataset saved successfully")