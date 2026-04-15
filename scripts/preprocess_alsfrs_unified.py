import pandas as pd

# Load Raw ALSFRS
df = pd.read_csv("data/Raw-data/F_PROACT_ALSFRS.csv")


df['Q10_Unified'] = df['Q10_Respiratory'].fillna(df.get('R_1_Dyspnea'))


q5_cols = ['Q5a_Cutting_without_Gastrostomy', 'Q5b_Cutting_with_Gastrostomy']
df['Q5_Unified'] = df.get('Q5_Cutting') 
for col in q5_cols:
    if col in df.columns:
        if 'Q5_Unified' in df.columns:
            df['Q5_Unified'] = df['Q5_Unified'].fillna(df[col])
        else:
            df['Q5_Unified'] = df[col]


questions = [
    'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing', 'Q4_Handwriting', 'Q5_Unified', 
    'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed', 'Q8_Walking', 'Q9_Climbing_Stairs', 'Q10_Unified'
]


df['ALSFRS_Total_Unified'] = df[questions].sum(axis=1)


df.to_csv("data/Preprocessed-data/ALSFRS_unified.csv", index=False)
print("Unified ALSFRS created. This cohort will help you reach ~2,921 patients.")