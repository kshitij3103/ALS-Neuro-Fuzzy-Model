import pandas as pd
import os

file_path = 'data/raw-data/F_PROACT_RILUZOLE.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    
    # 1. Exact Column Names
    # Identifying the ID and the indicator for Riluzole use
    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
    # 2. Check for "Yes/No" or "Binary" indicators
    # Looking for columns like 'Subject_used_Riluzole'
    print("\n[2] SAMPLE DATA:")
    print(df.head())
    
    # 3. Missing Values Percentage
    print("\n[3] MISSING VALUES PERCENTAGE:")
    print((df.isnull().sum() / len(df)) * 100)
    
    # 4. Record Count Check
    print(f"\nTotal Records: {len(df)}")
    print(f"Unique Patients: {df['subject_id'].nunique()}")
else:
    print(f"Error: {file_path} not found.")