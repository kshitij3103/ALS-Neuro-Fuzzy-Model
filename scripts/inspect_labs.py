import pandas as pd
import os

file_path = 'data/raw-data/F_PROACT_LABS.csv'

if os.path.exists(file_path):
    # Note: Labs can be huge. If it takes too long to load, use nrows=100000
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    
    # 1. Exact Column Names
    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
    # 2. Lab Structure Check
    # Is it "Wide" (one column per test) or "Long" (one 'Test_Name' column)?
    test_cols = [c for c in df.columns if 'test' in c.lower()]
    print(f"\n[2] TEST-RELATED COLUMNS: {test_cols}")
    
    # 3. Identify Time and Result Columns
    res_cols = [c for c in df.columns if any(x in c.lower() for x in ['result', 'delta', 'value', 'unit'])]
    print(f"\n[3] RESULT & TIME COLUMNS: {res_cols}")
    
    # 4. Longitudinal Check
    unique_patients = df['subject_id'].nunique()
    print(f"\nTotal Records: {len(df)}")
    print(f"Unique Patients: {unique_patients}")
    
    # 5. Missing Values Percentage (Top 15)
    print("\n[5] MISSING VALUES PERCENTAGE:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.sort_values(ascending=False).head(15))
    
    # 6. Sample Data
    print("\n[6] SAMPLE DATA (First 5 rows):")
    print(df.head())
else:
    print(f"Error: {file_path} not found.")