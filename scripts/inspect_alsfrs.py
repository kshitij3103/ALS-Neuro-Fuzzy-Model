import pandas as pd
import os

# Define the path based on your folder structure
file_path = 'data/raw-data/F_PROACT_ALSFRS.csv'

if os.path.exists(file_path):
    # Load the full table
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    print(f"Total Rows: {len(df)}")
    
    # 1. EXACT COLUMN NAMES (This identifies if it's 'subject_id', 'SubjectID', etc.)
    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
    # 2. MISSING VALUES PERCENTAGE (The 30% rule from the paper)
    print("\n[2] MISSING VALUES PERCENTAGE (Top 15):")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.sort_values(ascending=False).head(15))
    
    # 3. SAMPLE DATA & INFO
    print("\n[3] DATA INFO:")
    print(df.info())
    print("\n[4] SAMPLE DATA (First 5 rows):")
    print(df.head())
else:
    print(f"Error: {file_path} not found. Please check your directory.")