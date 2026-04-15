import pandas as pd
import os


file_path = 'data/raw-data/F_PROACT_ALSFRS.csv'

if os.path.exists(file_path):
  
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    print(f"Total Rows: {len(df)}")
    

    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
   
    print("\n[2] MISSING VALUES PERCENTAGE (Top 15):")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.sort_values(ascending=False).head(15))
    

    print("\n[3] DATA INFO:")
    print(df.info())
    print("\n[4] SAMPLE DATA (First 5 rows):")
    print(df.head())
else:
    print(f"Error: {file_path} not found. Please check your directory.")