# muscle_cleaning.py
# Clean and aggregate muscle strength dataset

import pandas as pd

print("Libraries loaded")

# Load dataset
muscle = pd.read_csv("Original-Data/F_PROACT_MUSCLESTRENGTH.csv")

print("\nDataset loaded")
print("Shape:", muscle.shape)

# Convert Test_Result to numeric
muscle["Test_Result"] = pd.to_numeric(muscle["Test_Result"], errors="coerce")

# Check missing values
print("\nMissing values:")
print(muscle.isnull().sum())

# Aggregate mean muscle strength per patient
muscle_mean = muscle.groupby("subject_id")["Test_Result"].mean().reset_index()

# Rename column
muscle_mean.rename(columns={"Test_Result": "Muscle_Strength_Mean"}, inplace=True)

print("\nDataset after grouping:")
print(muscle_mean.shape)

# Save cleaned dataset
muscle_mean.to_csv("PreProcessed-Data/muscle_cleaned.csv", index=False)

print("\nMuscle strength dataset saved successfully")