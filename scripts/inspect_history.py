import pandas as pd
import os

file_path = 'data/raw-data/F_PROACT_ALSHISTORY.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    
    # 1. Exact Column Names
    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
    # 2. Check for Onset Features
    # The paper looks for 'site of onset' and 'time of onset'
    onset_cols = [c for c in df.columns if 'onset' in c.lower() or 'site' in c.lower() or 'delta' in c.lower()]
    print(f"\n[2] POTENTIAL ONSET FEATURES: {onset_cols}")
    
    # 3. Missing Values Percentage
    print("\n[3] MISSING VALUES PERCENTAGE:")
    print((df.isnull().sum() / len(df)) * 100)
    
    # 4. Sample Data
    print("\n[4] SAMPLE DATA:")
    print(df.head())
else:
    print(f"Error: {file_path} not found.")