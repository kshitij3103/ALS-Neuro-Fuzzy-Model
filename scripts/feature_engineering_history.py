import pandas as pd

INPUT_PATH = "data/Raw-data/F_PROACT_ALSHISTORY.csv"
OUTPUT_PATH = "data/Preprocessed-data/ALSHISTORY_features.csv"

df = pd.read_csv(INPUT_PATH)


df["Onset_Duration"] = df["Onset_Delta"].abs()


df = df[["subject_id", "Onset_Duration"]].drop_duplicates(subset="subject_id")

df.to_csv(OUTPUT_PATH, index=False)
print(f"ALSHISTORY features saved. Final Shape: {df.shape} (1 feature + ID)")