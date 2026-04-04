import pandas as pd
import os

file_path = 'data/raw-data/F_PROACT_DEMOGRAPHICS.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    
    # 1. Exact Column Names
    # Confirming the ID column and identifying Age/Sex/Race
    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
    # 2. Key Demographic Features
    # Looking for Age, Sex, Race, or Gender
    demo_cols = [c for c in df.columns if any(x in c.lower() for x in ['age', 'sex', 'race', 'gender', 'ethnicity'])]
    print(f"\n[2] KEY DEMOGRAPHIC FEATURES: {demo_cols}")
    
    # 3. Missing Values Percentage
    print("\n[3] MISSING VALUES PERCENTAGE:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.sort_values(ascending=False))
    
    # 4. Record Count Check
    # Is it truly one record per patient?
    print(f"\nTotal Records: {len(df)}")
    print(f"Unique Patients: {df['subject_id'].nunique() if 'subject_id' in df.columns else 'ID Col Missing'}")
    
    # 5. Sample Data
    print("\n[5] SAMPLE DATA (First 5 rows):")
    print(df.head())
else:
    print(f"Error: {file_path} not found.")