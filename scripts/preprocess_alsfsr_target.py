import pandas as pd

INPUT_PATH = "data/Preprocessed-data/ALSFRS_unified.csv"
OUTPUT_PATH = "data/Preprocessed-data/ALSFRS_target.csv"

alsfrs = pd.read_csv(INPUT_PATH)

def compute_slope(group):
    group = group.sort_values("ALSFRS_Delta")
    
    if len(group[group["ALSFRS_Delta"] <= 90]) == 0: return None
    
    after_3 = group[group["ALSFRS_Delta"] >= 90]
    after_12 = group[group["ALSFRS_Delta"] >= 365]

    if len(after_3) == 0 or len(after_12) == 0: return None

    t1_row, t2_row = after_3.iloc[0], after_12.iloc[0]
    
   
    dt = (t2_row["ALSFRS_Delta"] - t1_row["ALSFRS_Delta"]) / 30.44
    if dt <= 0: return None
    
    return (t2_row["ALSFRS_Total_Unified"] - t1_row["ALSFRS_Total_Unified"]) / dt

target_df = alsfrs.groupby("subject_id").apply(compute_slope).dropna().reset_index()
target_df.columns = ["subject_id", "target_slope"]
target_df.to_csv(OUTPUT_PATH, index=False)
print(f"Target dataset saved for {len(target_df)} patients.")