import pandas as pd
import os

file_path = 'data/raw-data/F_PROACT_DEMOGRAPHICS.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    
   
    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
    
    #  Age, Sex, Race, or Gender
    demo_cols = [c for c in df.columns if any(x in c.lower() for x in ['age', 'sex', 'race', 'gender', 'ethnicity'])]
    print(f"\n[2] KEY DEMOGRAPHIC FEATURES: {demo_cols}")
    
    #  Missing Values Percentage
    print("\n[3] MISSING VALUES PERCENTAGE:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.sort_values(ascending=False))
    
   
    print(f"\nTotal Records: {len(df)}")
    print(f"Unique Patients: {df['subject_id'].nunique() if 'subject_id' in df.columns else 'ID Col Missing'}")
    
    
    print("\n[5] SAMPLE DATA (First 5 rows):")
    print(df.head())
else:
    print(f"Error: {file_path} not found.")