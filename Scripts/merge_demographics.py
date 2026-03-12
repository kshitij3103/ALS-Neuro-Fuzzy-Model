# Scripts/merge_demographics.py

import pandas as pd

print("Libraries loaded")

# -------------------------
# Load datasets
# -------------------------

als = pd.read_csv("PreProcessed-Data/alsfrs_cleaned.csv")
demo = pd.read_csv("Original-Data/F_PROACT_DEMOGRAPHICS.csv")

print("\nALSFRS shape:", als.shape)
print("Demographics shape:", demo.shape)

# -------------------------
# Select useful columns
# -------------------------

demo = demo[[
    "subject_id",
    "Age",
    "Sex"
]]

# -------------------------
# Encode Sex
# -------------------------

demo["Sex"] = demo["Sex"].map({
    "Male": 1,
    "Female": 0
})

# -------------------------
# Fill missing Age
# -------------------------

demo["Age"] = demo["Age"].fillna(demo["Age"].median())

demo["Sex"] = demo["Sex"].fillna(demo["Sex"].mode()[0])

# -------------------------
# Merge datasets
# -------------------------

df = pd.merge(
    als,
    demo,
    on="subject_id",
    how="inner"
)

print("\nMerged dataset shape:", df.shape)

# -------------------------
# Save merged dataset
# -------------------------

df.to_csv("PreProcessed-Data/alsfrs_demographics.csv", index=False)

print("\nMerged dataset saved")