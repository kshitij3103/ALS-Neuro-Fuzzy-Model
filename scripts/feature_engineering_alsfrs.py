import pandas as pd
import numpy as np

# Correct Input: Unified file which has both Original and Revised patients
INPUT_PATH = "data/Preprocessed-data/ALSFRS_unified.csv" 
OUTPUT_PATH = "data/Preprocessed-data/ALSFRS_features.csv"

alsfrs = pd.read_csv(INPUT_PATH)

# Filter first 3 months (0-90 days)
alsfrs = alsfrs[alsfrs["ALSFRS_Delta"] <= 90]

# Questions + Unified Total Score as features
question_cols = [
    "Q1_Speech", "Q2_Salivation", "Q3_Swallowing", "Q4_Handwriting", "Q5_Unified",
    "Q6_Dressing_and_Hygiene", "Q7_Turning_in_Bed", "Q8_Walking", "Q9_Climbing_Stairs", "Q10_Unified"
]
# Yahan 'ALSFRS_Total_Unified' add kiya gaya hai
longitudinal_features = ["ALSFRS_Total_Unified"] + question_cols

def summarize_feature(values, times):
    features = {}
    features["min"] = values.min()
    features["max"] = values.max()
    features["median"] = values.median()
    features["std"] = values.std() if len(values) > 1 else 0
    features["first"] = values.iloc[0]
    features["last"] = values.iloc[-1]
    if len(values) > 1:
        dt = times.iloc[-1] - times.iloc[0]
        features["slope"] = (values.iloc[-1] - values.iloc[0]) / (dt / 30) if dt > 0 else np.nan
    else:
        features["slope"] = np.nan
    return features

def compute_features(group):
    group = group.sort_values("ALSFRS_Delta")
    features = {}
    for col in longitudinal_features:
        summary = summarize_feature(group[col], group["ALSFRS_Delta"])
        for k, v in summary.items():
            features[f"{col}_{k}"] = v
    return pd.Series(features)

print("Processing ALSFRS features (including Total Score)...")
features_df = alsfrs.groupby("subject_id").apply(compute_features).reset_index()
features_df.to_csv(OUTPUT_PATH, index=False)
print(f"ALSFRS features saved with {features_df.shape[1]} columns.")