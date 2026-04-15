import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import os
import sys
from paper_dl_models import Paper_FFNN, Paper_CNN_Hybrid, Paper_RNN_Hybrid
from torch.utils.data import Dataset, DataLoader

# ==========================================
# CONFIG
# ==========================================
STATIC_DATA_PATH  = 'data/Preprocessed-data/MASTER_RAW_DATA.csv'
DYNAMIC_DATA_PATH = 'data/Preprocessed-data/ALSFRS_unified.csv'
TARGET_COL        = 'target_slope'
SUBJECT_COL       = 'subject_id'
N_FOLDS           = 5
N_EPOCHS          = 30
BATCH_SIZE        = 32
LR                = 0.0005  # Reduced LR 
WEIGHT_DECAY      = 1e-4    # L2 regularization
MAX_VISITS        = 5
TOP_K_FEATURES    = 30      # Base paper optimized FFNN with top 30
OUTPUT_DIR        = 'analysis/paper_dl_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
print("Loading data...")
df_static = pd.read_csv(STATIC_DATA_PATH)
df_dynamic = pd.read_csv(DYNAMIC_DATA_PATH)

drop_cols = [c for c in df_static.columns if 'Q1' in c or 'Q2' in c or 'Q3' in c or 'Q4' in c 
             or 'Q5' in c or 'Q6' in c or 'Q7' in c or 'Q8' in c or 'Q9' in c or 'Q10' in c 
             or 'ALSFRS' in c]
static_features_only = df_static.drop(columns=drop_cols)

dynamic_cols = ['Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing', 'Q4_Handwriting', 
                'Q5_Unified', 'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed', 
                'Q8_Walking', 'Q9_Climbing_Stairs', 'Q10_Unified', 'ALSFRS_Delta']

df_dynamic = df_dynamic[['subject_id'] + dynamic_cols]

print("Building dynamic 11x5 matrices for CNN/RNN...")
patient_dynamic_data = {}
gb = df_dynamic.groupby('subject_id')
for subject, group in gb:
    early_group = group[group['ALSFRS_Delta'] <= 90].copy()
    if len(early_group) == 0:
        early_group = group.head(1).copy()
        
    early_group = early_group.sort_values('ALSFRS_Delta').head(MAX_VISITS)
    early_group = early_group.ffill().bfill()
    
    visits = early_group[dynamic_cols].values # (n_visits, 11)
    
    if len(visits) < MAX_VISITS:
        pad_size = MAX_VISITS - len(visits)
        if len(visits) > 0:
            last_row = visits[-1].reshape(1, -1)
        else:
            last_row = np.zeros((1, 11))
        padding = np.repeat(last_row, pad_size, axis=0)
        visits = np.vstack([visits, padding])
    
    # Transpose to (11 questions, 5 visits)
    matrix_11x5 = visits.T
    matrix_11x5 = np.nan_to_num(matrix_11x5, nan=0.0)
    patient_dynamic_data[subject] = matrix_11x5

static_subjects = set(static_features_only['subject_id'].values)
dynamic_subjects = set(patient_dynamic_data.keys())
common_subjects = list(static_subjects.intersection(dynamic_subjects))

df_static_filtered = static_features_only[static_features_only['subject_id'].isin(common_subjects)].reset_index(drop=True)

y = df_static_filtered[TARGET_COL].values
groups = df_static_filtered[SUBJECT_COL].values
X_static_df = df_static_filtered.drop(columns=[SUBJECT_COL, TARGET_COL])

X_static_df = X_static_df.fillna(X_static_df.median())
X_static_raw = X_static_df.values

X_dynamic_raw = np.array([patient_dynamic_data[subj] for subj in groups]) # shape: (N, 11, 5)

class ALSDataset(Dataset):
    def __init__(self, static_data, dynamic_data, targets):
        self.static_data = torch.FloatTensor(static_data)
        self.dynamic_data = torch.FloatTensor(dynamic_data).unsqueeze(1)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.static_data[idx], self.dynamic_data[idx], self.targets[idx]

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def pcc(y_true, y_pred):
    if len(np.unique(y_pred)) < 2: return 0.0
    return pearsonr(y_true.flatten(), y_pred.flatten())[0]

models_to_test = {
    "FFNN": Paper_FFNN,
    "CNN_Hybrid": Paper_CNN_Hybrid,
    "RNN_Hybrid": Paper_RNN_Hybrid
}

results = []

print("\n===========================================")
print("STARTING OPTIMIZED 5-FOLD CV DL EVALUATION")
print("===========================================\n")

for model_name, model_class in models_to_test.items():
    print(f"--- Training {model_name} ---")
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    fold_rmsd = []
    fold_pcc = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_static_raw, y, groups)):
        
        # 1. Base Split
        X_train_stat, X_test_stat = X_static_raw[train_idx], X_static_raw[test_idx]
        X_train_dyn, X_test_dyn = X_dynamic_raw[train_idx], X_dynamic_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 2. Extract Top 30 Static Features via Random Forest 
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_stat, y_train)
        top_indices = np.argsort(rf.feature_importances_)[-TOP_K_FEATURES:]
        
        X_train_stat = X_train_stat[:, top_indices]
        X_test_stat = X_test_stat[:, top_indices]
        
        # 3. Scale Static Features 
        scaler_stat = StandardScaler()
        X_train_stat_scaled = scaler_stat.fit_transform(X_train_stat)
        X_test_stat_scaled = scaler_stat.transform(X_test_stat)
        
        # 4. Scale Dynamic Features (Unroll -> Scale -> Reshape)
        N_train, channels, time_steps = X_train_dyn.shape
        N_test = X_test_dyn.shape[0]
        
        X_train_dyn_flat = X_train_dyn.reshape(N_train, -1)
        X_test_dyn_flat = X_test_dyn.reshape(N_test, -1)
        
        scaler_dyn = StandardScaler()
        X_train_dyn_flat_scaled = scaler_dyn.fit_transform(X_train_dyn_flat)
        X_test_dyn_flat_scaled = scaler_dyn.transform(X_test_dyn_flat)
        
        X_train_dyn_scaled = X_train_dyn_flat_scaled.reshape(N_train, channels, time_steps)
        X_test_dyn_scaled = X_test_dyn_flat_scaled.reshape(N_test, channels, time_steps)

        # 5. Loaders
        train_dataset = ALSDataset(X_train_stat_scaled, X_train_dyn_scaled, y_train)
        test_dataset = ALSDataset(X_test_stat_scaled, X_test_dyn_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 6. Init Model
        model = model_class(num_static_features=TOP_K_FEATURES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()
        
        # Training Routine
        model.train()
        for epoch in range(N_EPOCHS):
            for stat_b, dyn_b, target_b in train_loader:
                optimizer.zero_grad()
                pred = model(stat_b, dyn_b)
                loss = criterion(pred, target_b)
                loss.backward()
                optimizer.step()
        
        # Validation Routine
        model.eval()
        fold_preds = []
        fold_trues = []
        with torch.no_grad():
            for stat_b, dyn_b, target_b in test_loader:
                pred = model(stat_b, dyn_b)
                fold_preds.extend(pred.cpu().numpy())
                fold_trues.extend(target_b.cpu().numpy())
        
        fold_preds = np.array(fold_preds)
        fold_trues = np.array(fold_trues)
        
        current_rmsd = rmse(fold_trues, fold_preds)
        current_pcc = pcc(fold_trues, fold_preds)
        
        print(f"  Fold {fold+1}/{N_FOLDS} | RMSD: {current_rmsd:.4f} | PCC: {current_pcc:.4f}")
        
        fold_rmsd.append(current_rmsd)
        fold_pcc.append(current_pcc)
    
    avg_rmsd = np.mean(fold_rmsd)
    avg_pcc = np.mean(fold_pcc)
    print(f"\n[!] Final {model_name} Metrics => Average RMSD: {avg_rmsd:.4f} | Average PCC: {avg_pcc:.4f}\n")
    
    results.append({
        'Model': model_name,
        'Mean_RMSD': avg_rmsd,
        'Mean_PCC': avg_pcc
    })

pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'paper_dl_results.csv'), index=False)
print(f"Saved optimized paper evaluation results to: {OUTPUT_DIR}/paper_dl_results.csv")
