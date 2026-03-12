# Scripts/demographics_analysis.py

import pandas as pd

print("Libraries loaded")

# Load demographics dataset
demo = pd.read_csv("Original-Data/F_PROACT_DEMOGRAPHICS.csv")

print("\nDataset loaded")

# Shape
print("\nShape:", demo.shape)

# Columns
print("\nColumns:")
print(demo.columns)

# First rows
print("\nFirst 5 rows:")
print(demo.head())

# Missing values
print("\nMissing values:")
print(demo.isnull().sum())
# -----------------------------
# Select useful columns
# -----------------------------

demo = demo[[
    "subject_id",
    "Age",
    "Sex",
    "Ethnicity",
    
]]

print("\nSelected columns:")
print(demo.head())
# -----------------------------
# Encode Sex column
# -----------------------------

demo["Sex"] = demo["Sex"].map({
    "Male": 1,
    "Female": 0
})
# -----------------------------
# Fill missing Age values
# -----------------------------

demo["Age"] = demo["Age"].fillna(demo["Age"].median())
print("\nMissing values after cleaning:")
print(demo.isnull().sum())
# Fill missing Sex
demo["Sex"] = demo["Sex"].fillna(demo["Sex"].mode()[0])



print("\nMissing values after final cleaning:")
print(demo.isnull().sum())