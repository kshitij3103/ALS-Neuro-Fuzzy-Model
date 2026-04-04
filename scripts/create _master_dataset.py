import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import os

print("--- PHASE 4: FINAL MASTER MERGE (STRICT PROTECTION ENABLED) ---")

# 1. Target List Load karein (2,921+ patients criteria)
target_path = 'data/Preprocessed-data/ALSFRS_target.csv'
if not os.path.exists(target_path):
    print(f"ERROR: {target_path} nahi mili. Pehle target slope script run karein.")
    exit()

target_df = pd.read_csv(target_path)
master_df = target_df[['subject_id', 'target_slope']].drop_duplicates(subset=['subject_id'])

# 2. Saari 7 Feature Tables ki list
feature_files = [
    'ALSFRS_features.csv',      # 10 Questions + Total (70+ features)
    'VitalSigns_features.csv',  # 5 Variables (35 features)
    'LABS_features.csv',        # 24 Selected Labs (168 features)
    'ALSHISTORY_features.csv',   # Onset Duration (Critical)
    'FVC_features.csv',         # Respiratory (Top Predictor)
    'STATIC_features.csv' # Age, Sex, Race
    
]

print(f"Merging features for {len(master_df)} patients...")

for f_name in feature_files:
    path = f"data/Preprocessed-data/{f_name}"
    if os.path.exists(path):
        feat_df = pd.read_csv(path).drop_duplicates(subset=['subject_id'])
        master_df = master_df.merge(feat_df, on='subject_id', how='left')
        print(f"Merged {f_name}")
    else:
        print(f"Warning: {f_name}  (File not found).")

# 3. Features aur Target alag karein
ids = master_df['subject_id']
y = master_df['target_slope']
X = master_df.drop(columns=['subject_id', 'target_slope'])

# Force to numeric (Safety check for clinical data)
X = X.apply(pd.to_numeric, errors='coerce')

# ========================================================
# 4. STRICT PROTECTION LOGIC (ALSFRS, FVC, ONSET)
# "Domain knowledge overrides the 30% missingness rule"
# ========================================================

# In clinical pillars ke har ek score (min, max, median, std, first, last, slope) ko protect karna hai
protected_categories = [
    'ALSFRS_Total_Unified', # Global progression
    'Q1_', 'Q2_', 'Q3_', 'Q4_', 'Q5_', 'Q6_', 'Q7_', 'Q8_', 'Q9_', 'Q10_', # Functional questions
    'FVC_', # Breathing capacity
    'Onset_' # Time since symptoms (Highest importance)
]

# Protected columns identify karein
protected_cols = [col for col in X.columns if any(p in col for p in protected_categories)]

# 30% Missingness Filter (Standard for Labs/Vitals)
missing_ratio = X.isnull().mean()
standard_keep = missing_ratio[missing_ratio <= 0.30].index.tolist()

# Final Features = (Standard 30% Rule) OR (Protected Clinical Pillar)
final_cols = list(set(standard_keep) | set(protected_cols))
X_final = X[final_cols]

print(f"\nInitial total features: {X.shape[1]}")
print(f"Features kept after filtering & protection: {X_final.shape[1]}")
print(f"Protected features 'rescued' from drop: {len(set(protected_cols) - set(standard_keep))}")

# ========================================================
# 5. IMPUTATION & SCALING
# ========================================================
scaler = MinMaxScaler()

# VERSION A: MASTER_MEAN_DATA (Paper's approach)
print("\nProcessing MASTER_MEAN_DATA...")
X_mean = X_final.fillna(X_final.mean())
X_scaled_mean = scaler.fit_transform(X_mean)
final_mean = pd.DataFrame(X_scaled_mean, columns=X_final.columns)
final_mean.insert(0, 'subject_id', ids.values)
final_mean['target_slope'] = y.values
final_mean.to_csv('data/Preprocessed-data/MASTER_MEAN_DATA.csv', index=False)

# VERSION B: MASTER_KNN_DATA (Enhanced Imputation)
print("Processing MASTER_KNN_DATA (K=5)...")
imputer = KNNImputer(n_neighbors=5)
X_knn = imputer.fit_transform(X_final)
X_scaled_knn = scaler.fit_transform(X_knn)
final_knn = pd.DataFrame(X_scaled_knn, columns=X_final.columns)
final_knn.insert(0, 'subject_id', ids.values)
final_knn['target_slope'] = y.values
final_knn.to_csv('data/Preprocessed-data/MASTER_KNN_DATA.csv', index=False)

print("\n--- MASTER PIPELINE COMPLETE ---")
print(f"Final Feature Count: {X_final.shape[1]}")
print(f"Final Patient Count: {len(master_df)}")
print("Files saved: MASTER_MEAN_DATA.csv and MASTER_KNN_DATA.csv")