# scripts/data_loader.py

# STEP 1: Import pandas library
import pandas as pd

print("Libraries loaded successfully")

# STEP 2: Load ALSFRS dataset
alsfrs = pd.read_csv("Original-Data/F_PROACT_ALSFRS.csv")

print("ALSFRS dataset loaded")

# STEP 3: Check dataset shape
print("ALSFRS dataset shape:", alsfrs.shape)
# STEP 4: Show column names
print("\nALSFRS Columns:")
print(alsfrs.columns)
# STEP 5: Check unique patients
print("\nUnique patients:", alsfrs["subject_id"].nunique())
# STEP 6: Keep latest visit per patient

alsfrs_latest = alsfrs.groupby("subject_id").last().reset_index()

print("\nDataset after selecting latest visit:")
print("Shape:", alsfrs_latest.shape)
# STEP 7: Check missing values

print("\nMissing values per column:")
print(alsfrs_latest.isnull().sum())
# STEP 8: Drop columns with excessive missing values

drop_cols = [
    "Q5b_Cutting_with_Gastrostomy",
    "Mode_of_Administration",
    "ALSFRS_Responded_By",
    "ALSFRS_Total"
]

alsfrs_latest = alsfrs_latest.drop(columns=drop_cols)

print("\nDataset shape after dropping useless columns:")
print(alsfrs_latest.shape)
# STEP 9: Fill missing values using median

alsfrs_latest = alsfrs_latest.fillna(alsfrs_latest.median(numeric_only=True))

print("\nMissing values after imputation:")
print(alsfrs_latest.isnull().sum())
alsfrs_latest.to_csv("PreProcessed-Data/alsfrs_cleaned.csv", index=False)

print("\nProcessed dataset saved successfully")