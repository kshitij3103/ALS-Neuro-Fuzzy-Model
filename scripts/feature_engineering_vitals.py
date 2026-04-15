import pandas as pd
import numpy as np

INPUT_PATH = "data/Raw-data/F_PROACT_VITALSIGNS.csv"
OUTPUT_PATH = "data/Preprocessed-data/VitalSigns_features.csv"

vital = pd.read_csv(INPUT_PATH, low_memory=False)


variables = ["Blood_Pressure_Diastolic", "Blood_Pressure_Systolic", "Pulse", "Respiratory_Rate", "Weight"]
cols = ["subject_id", "Vital_Signs_Delta"] + variables
vital = vital[cols]

# Force numeric and filter unrealistic values
for col in variables:
    vital[col] = pd.to_numeric(vital[col], errors="coerce")

vital.loc[(vital["Weight"] < 30) | (vital["Weight"] > 200), "Weight"] = np.nan
vital.loc[(vital["Blood_Pressure_Systolic"] < 70) | (vital["Blood_Pressure_Systolic"] > 220), "Blood_Pressure_Systolic"] = np.nan
vital.loc[(vital["Blood_Pressure_Diastolic"] < 40) | (vital["Blood_Pressure_Diastolic"] > 140), "Blood_Pressure_Diastolic"] = np.nan

vital = vital[(vital["Vital_Signs_Delta"] >= 0) & (vital["Vital_Signs_Delta"] <= 90)]

rows = []
for pid, group in vital.groupby("subject_id"):
    group = group.sort_values("Vital_Signs_Delta")
    features = {"subject_id": pid}
    for var in variables:
        values = group[var].dropna()
        if len(values) == 0: continue

        features[f"{var}_min"] = values.min()
        features[f"{var}_max"] = values.max()
        features[f"{var}_median"] = values.median()
        features[f"{var}_std"] = values.std() if len(values) > 1 else 0
        features[f"{var}_first"] = values.iloc[0]
        features[f"{var}_last"] = values.iloc[-1]

        if len(values) > 1:
            dt = group.loc[values.index, "Vital_Signs_Delta"].iloc[-1] - group.loc[values.index, "Vital_Signs_Delta"].iloc[0]
            features[f"{var}_slope"] = (values.iloc[-1] - values.iloc[0]) / (dt / 30) if dt >= 20 else np.nan
        else:
            features[f"{var}_slope"] = np.nan
    rows.append(features)

features_df = pd.DataFrame(rows)
# Clean some outliers
features_df.loc[(features_df["Weight_slope"] < -5) | (features_df["Weight_slope"] > 5), "Weight_slope"] = np.nan
features_df.to_csv(OUTPUT_PATH, index=False)
print(f"VitalSigns features saved. Final Shape: {features_df.shape} (35 features + ID)")