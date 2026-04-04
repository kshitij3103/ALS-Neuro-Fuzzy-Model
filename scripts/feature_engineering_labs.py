import pandas as pd
import numpy as np
import os

# ===============================
# Paths
# ===============================
INPUT_PATH = "data/Raw-data/F_PROACT_LABS.csv"
OUTPUT_DIR = "data/Preprocessed-data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "LABS_features.csv")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ===============================
# The 24 Lab Tests from Base Paper
# ===============================
# These specific tests are used to reach the 274 feature count
KEEP_LABS = [
    'Albumin', 'Alkaline Phosphatase', 'ALT (SGPT)', 'AST (SGOT)', 'Bilirubin (Total)', 
    'Blood Urea Nitrogen (BUN)', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 
    'Hematocrit', 'Hemoglobin', 'Lymphocytes', 'Monocytes', 'Neutrophils', 
    'Phosphorus', 'Platelets', 'Potassium', 'Red Blood Cells (RBC)', 'Sodium', 
    'Total Cholesterol', 'Total Protein', 'Uric Acid', 'White Blood Cell (WBC)'
]

# ===============================
# Load and Clean
# ===============================
print("Loading Labs data...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Convert results to numeric (forces strings to NaN)
df["Test_Result"] = pd.to_numeric(df["Test_Result"], errors="coerce")
df = df.dropna(subset=["Test_Result"])

# Filter for the 24 specific tests
df = df[df['Test_Name'].isin(KEEP_LABS)]

# Filter for the 3-month window (0-90 days)
df_3m = df[(df["Laboratory_Delta"] >= 0) & (df["Laboratory_Delta"] <= 90)].copy()

print(f"Processing {df_3m['Test_Name'].nunique()} tests for {df_3m['subject_id'].nunique()} patients...")

# ===============================
# Feature Extraction (7-Score Vector)
# ===============================
rows = []
# Group by patient and test to calculate stats
for (pid, test), group in df_3m.groupby(["subject_id", "Test_Name"]):
    group = group.sort_values("Laboratory_Delta")
    v, t = group["Test_Result"], group["Laboratory_Delta"]
    
    dt = t.iloc[-1] - t.iloc[0]
    
    features = {
        "subject_id": pid,
        "Test_Name": test,
        "min": v.min(),
        "max": v.max(),
        "median": v.median(),
        "std": v.std() if len(v) > 1 else 0,
        "first": v.iloc[0],
        "last": v.iloc[-1],
    }
    
    # Calculate monthly normalized slope
    if len(v) > 1 and dt > 0:
        features["slope"] = (v.iloc[-1] - v.iloc[0]) / (dt / 30)
    else:
        features["slope"] = np.nan
        
    rows.append(features)

# ===============================
# Shape Transformation
# ===============================
features_df = pd.DataFrame(rows)

# Pivot from "Long" to "Wide" format (One row per patient)
print("Pivoting to wide format...")
pivot_df = features_df.pivot_table(index="subject_id", columns="Test_Name")

# Flatten multi-level column names: (stat, test) -> test_stat
pivot_df.columns = [f"{test}_{stat}" for stat, test in pivot_df.columns]
pivot_df = pivot_df.reset_index()

# ===============================
# Save Final Lab Matrix
# ===============================
pivot_df.to_csv(OUTPUT_PATH, index=False)
print(f"LABS features saved. Final Shape: {pivot_df.shape}")