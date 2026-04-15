import pandas as pd
import numpy as np
import os


INPUT_PATH = "data/Raw-data/F_PROACT_LABS.csv"
OUTPUT_DIR = "data/Preprocessed-data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "LABS_features.csv")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


KEEP_LABS = [
    'Albumin', 'Alkaline Phosphatase', 'ALT (SGPT)', 'AST (SGOT)', 'Bilirubin (Total)', 
    'Blood Urea Nitrogen (BUN)', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 
    'Hematocrit', 'Hemoglobin', 'Lymphocytes', 'Monocytes', 'Neutrophils', 
    'Phosphorus', 'Platelets', 'Potassium', 'Red Blood Cells (RBC)', 'Sodium', 
    'Total Cholesterol', 'Total Protein', 'Uric Acid', 'White Blood Cell (WBC)'
]


print("Loading Labs data...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

df["Test_Result"] = pd.to_numeric(df["Test_Result"], errors="coerce")
df = df.dropna(subset=["Test_Result"])


df = df[df['Test_Name'].isin(KEEP_LABS)]

df_3m = df[(df["Laboratory_Delta"] >= 0) & (df["Laboratory_Delta"] <= 90)].copy()

print(f"Processing {df_3m['Test_Name'].nunique()} tests for {df_3m['subject_id'].nunique()} patients...")


rows = []

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
    

    if len(v) > 1 and dt > 0:
        features["slope"] = (v.iloc[-1] - v.iloc[0]) / (dt / 30)
    else:
        features["slope"] = np.nan
        
    rows.append(features)


features_df = pd.DataFrame(rows)


print("Pivoting to wide format...")
pivot_df = features_df.pivot_table(index="subject_id", columns="Test_Name")


pivot_df.columns = [f"{test}_{stat}" for stat, test in pivot_df.columns]
pivot_df = pivot_df.reset_index()


pivot_df.to_csv(OUTPUT_PATH, index=False)
print(f"LABS features saved. Final Shape: {pivot_df.shape}")