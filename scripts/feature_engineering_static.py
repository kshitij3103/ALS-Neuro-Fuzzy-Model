import pandas as pd

demo = pd.read_csv("data/raw-data/F_PROACT_DEMOGRAPHICS.csv")[["subject_id", "Age", "Sex"]]

rilu = pd.read_csv("data/raw-data/F_PROACT_RILUZOLE.csv")[["subject_id", "Subject_used_Riluzole"]]

# Merge all static info
static = demo.merge(rilu, on="subject_id", how="outer")

# Encode categories for the model
static["Sex"] = static["Sex"].map({"Male": 1, "Female": 0})
static["Subject_used_Riluzole"] = static["Subject_used_Riluzole"].map({"Yes": 1, "No": 0})

static.to_csv("data/preprocessed-data/STATIC_features.csv", index=False)
print("Static features saved.")