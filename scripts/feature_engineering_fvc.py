import pandas as pd
import numpy as np

df = pd.read_csv("data/raw-data/F_PROACT_FVC.csv")

# Rule: Use the maximum liter value from the 3 trials 
df["fvc_max"] = df[["Subject_Liters_Trial_1", "Subject_Liters_Trial_2", "Subject_Liters_Trial_3"]].max(axis=1)
df_3m = df[(df["Forced_Vital_Capacity_Delta"] >= 0) & (df["Forced_Vital_Capacity_Delta"] <= 90)]

def summarize_fvc(group):
    group = group.sort_values("Forced_Vital_Capacity_Delta")
    v, t = group["fvc_max"], group["Forced_Vital_Capacity_Delta"]
    dt = t.iloc[-1] - t.iloc[0]
    return pd.Series({
        "FVC_min": v.min(), "FVC_max": v.max(), "FVC_median": v.median(),
        "FVC_std": v.std() if len(v) > 1 else 0,
        "FVC_first": v.iloc[0], "FVC_last": v.iloc[-1],
        "FVC_slope": (v.iloc[-1] - v.iloc[0]) / (dt / 30) if dt > 0 else np.nan
    })

fvc_features = df_3m.groupby("subject_id").apply(summarize_fvc).reset_index()
fvc_features.to_csv("data/preprocessed-data/FVC_features.csv", index=False)
print("FVC features saved.")