import pandas as pd
import os

file_path = 'data/raw-data/F_PROACT_FVC.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"--- INSPECTING: {os.path.basename(file_path)} ---")
    
    
    print("\n[1] EXACT COLUMN NAMES:")
    print(df.columns.tolist())
    
   
    # for FVC, Liters, Percentage, and Delta/Day
    fvc_cols = [c for c in df.columns if any(x in c.lower() for x in ['fvc', 'liter', 'percent', 'delta', 'day'])]
    print(f"\n[2] KEY FVC FEATURES: {fvc_cols}")
    
   
    print("\n[3] MISSING VALUES PERCENTAGE:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.sort_values(ascending=False))
    
    
    unique_patients = df['subject_id'].nunique()
    print(f"\nTotal Records: {len(df)}")
    print(f"Unique Patients: {unique_patients}")
    print(f"Avg records per patient: {len(df)/unique_patients:.2f}")
    
    
    print("\n[5] SAMPLE DATA (First 5 rows):")
    print(df.head())
else:
    print(f"Error: {file_path} not found.")