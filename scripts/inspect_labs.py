import pandas as pd
import os

file_path = 'data/raw-data/F_PROACT_LABS.csv'

if os.path.exists(file_path):
   
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    

    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
    
    test_cols = [c for c in df.columns if 'test' in c.lower()]
    print(f"\n[2] TEST-RELATED COLUMNS: {test_cols}")
    
  
    res_cols = [c for c in df.columns if any(x in c.lower() for x in ['result', 'delta', 'value', 'unit'])]
    print(f"\n[3] RESULT & TIME COLUMNS: {res_cols}")
    
    # 4. Longitudinal Check
    unique_patients = df['subject_id'].nunique()
    print(f"\nTotal Records: {len(df)}")
    print(f"Unique Patients: {unique_patients}")
    
    # 5. Missing Values Percentage 
    print("\n[5] MISSING VALUES PERCENTAGE:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.sort_values(ascending=False).head(15))
    

    print("\n[6] SAMPLE DATA (First 5 rows):")
    print(df.head())
else:
    print(f"Error: {file_path} not found.")