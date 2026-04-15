import pandas as pd
import os

print("--- CREATING MASTER RAW DATASET (NO IMPUTATION) ---")


target_df = pd.read_csv('data/Preprocessed-data/ALSFRS_target.csv')
master_raw = target_df[['subject_id', 'target_slope']].drop_duplicates(subset=['subject_id'])

feature_files = [
    'ALSFRS_features.csv', 'VitalSigns_features.csv', 'LABS_features.csv', 
    'ALSHISTORY_features.csv', 'FVC_features.csv', 'STATIC_features.csv' 
]


for f_name in feature_files:
    path = f"data/Preprocessed-data/{f_name}"
    if os.path.exists(path):
        feat_df = pd.read_csv(path).drop_duplicates(subset=['subject_id'])
        master_raw = master_raw.merge(feat_df, on='subject_id', how='left')
        print(f"Merged {f_name}")


output_path = 'data/Preprocessed-data/MASTER_RAW_DATA.csv'
master_raw.to_csv(output_path, index=False)
print(f"\nSUCCESS: Raw dataset saved with {len(master_raw)} patients.")