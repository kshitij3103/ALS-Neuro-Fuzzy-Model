import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import os

# ==========================================
# CONFIG
# ==========================================
DATA_PATH      = 'data/Preprocessed-data/MASTER_RAW_DATA.csv'
SAVE_PATH      = 'data/Preprocessed-data/trained_anfis_ensemble.pth'
ELITE_5        = ['Onset_Duration', 'ALSFRS_Total_Unified_median',
                  'FVC_slope', 'Weight_slope', 'Q1_Speech_min']
TARGET_COL     = 'target_slope'
SUBJECT_COL    = 'subject_id'
N_FOLDS        = 5
N_EPOCHS       = 15
N_CLUSTERS     = 10         # Back to 10 — contradiction checker handles bad rules
LSE_LAMBDA     = 0.1        # Reduced from 2.0 — prevents consequent flattening
ENSEMBLE_SEEDS = [42, 52, 62]


# ==========================================
# HYBRID ANFIS ARCHITECTURE
# ==========================================
class ClusterANFIS(nn.Module):
    def __init__(self, num_inputs=5, num_clusters=10, initial_centers=None):
        super(ClusterANFIS, self).__init__()
        self.num_inputs   = num_inputs
        self.num_clusters = num_clusters
        self.centers = nn.Parameter(initial_centers.clone())
        self.sigmas  = nn.Parameter(torch.ones(num_clusters, num_inputs) * 1.5)
        self.register_buffer(
            'consequent_params',
            torch.zeros(num_clusters * (num_inputs + 1), 1)
        )

    def forward_premise(self, x):
        x_expanded = x.unsqueeze(1)
        s  = torch.nn.functional.softplus(self.sigmas) + 1e-5
        mu = torch.exp(-((x_expanded - self.centers) ** 2) / (2 * s ** 2))
        w  = torch.prod(mu, dim=2)
        return w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)

    def construct_design_matrix(self, x, w_norm):
        batch_size = x.shape[0]
        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)
        return (w_norm.unsqueeze(2) * x_ext.unsqueeze(1)).view(batch_size, -1)

    def forward(self, x):
        w_norm = self.forward_premise(x)
        A      = self.construct_design_matrix(x, w_norm)
        return torch.matmul(A, self.consequent_params)

    def lse_update(self, x, y, lam=LSE_LAMBDA):
        """
        Hybrid learning: least-squares update for consequent parameters.
        lam controls regularization — lower = more expressive consequents.
        """
        with torch.no_grad():
            w_norm = self.forward_premise(x)
            A      = self.construct_design_matrix(x, w_norm)
            I      = torch.eye(A.shape[1], device=A.device)
            AtA    = torch.matmul(A.T, A) + lam * I
            new_params = torch.linalg.solve(AtA, torch.matmul(A.T, y))
            self.consequent_params.copy_(new_params)


# ==========================================
# FOLD-LEVEL IMPUTATION
# ==========================================
def impute_with_train_median(X_tr: np.ndarray, X_te: np.ndarray):
    medians = np.nanmedian(X_tr, axis=0)
    for col in range(X_tr.shape[1]):
        X_tr[:, col] = np.where(np.isnan(X_tr[:, col]), medians[col], X_tr[:, col])
        X_te[:, col] = np.where(np.isnan(X_te[:, col]), medians[col], X_te[:, col])
    return X_tr, X_te


# ==========================================
# SINGLE ENSEMBLE MEMBER TRAINING
# ==========================================
def train_member(X_tr_scaled, y_tr_scaled, seed,
                 n_epochs=N_EPOCHS, n_clusters=N_CLUSTERS):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X_tr_scaled.numpy())
    init_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    model = ClusterANFIS(num_inputs=X_tr_scaled.shape[1],
                         num_clusters=n_clusters,
                         initial_centers=init_centers)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(n_epochs):
        model.lse_update(X_tr_scaled, y_tr_scaled)
        model.train()
        opt.zero_grad()
        loss = nn.MSELoss()(model(X_tr_scaled), y_tr_scaled)
        loss.backward()
        opt.step()

    model.eval()
    return model


# ==========================================
# MAIN VALIDATION LOOP
# ==========================================
def run_final_validation(k=N_FOLDS):
    data = pd.read_csv(DATA_PATH)

    missing = [c for c in ELITE_5 + [TARGET_COL, SUBJECT_COL] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X      = data[ELITE_5].values.astype(float)
    y      = data[TARGET_COL].values.reshape(-1, 1)
    groups = data[SUBJECT_COL].values

    gkf = GroupKFold(n_splits=k)
    fold_pccs, fold_rmsds = [], []
    best_pcc, best_fold_data = -np.inf, {}

    print(f"\nStarting {k}-Fold Validation")
    print(f"GroupKFold | {len(ENSEMBLE_SEEDS)} members | "
          f"{N_EPOCHS} Epochs | {N_CLUSTERS} Clusters | LSE λ={LSE_LAMBDA}")
    print("-" * 60)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr_raw, X_te_raw = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te         = y[train_idx], y[test_idx]

        X_tr_raw, X_te_raw = impute_with_train_median(X_tr_raw, X_te_raw)

        sc_x, sc_y  = StandardScaler(), StandardScaler()
        X_tr_scaled = torch.tensor(sc_x.fit_transform(X_tr_raw), dtype=torch.float32)
        y_tr_scaled = torch.tensor(sc_y.fit_transform(y_tr),     dtype=torch.float32)
        X_te_scaled = torch.tensor(sc_x.transform(X_te_raw),     dtype=torch.float32)

        member_models, member_preds = [], []
        for seed in ENSEMBLE_SEEDS:
            m = train_member(X_tr_scaled, y_tr_scaled, seed)
            member_models.append(m)
            with torch.no_grad():
                member_preds.append(m(X_te_scaled).numpy())

        y_pred = sc_y.inverse_transform(np.mean(member_preds, axis=0))
        p, _   = pearsonr(y_te.flatten(), y_pred.flatten())
        r      = np.sqrt(np.mean((y_te - y_pred) ** 2))

        fold_pccs.append(p)
        fold_rmsds.append(r)
        print(f"Fold {fold+1} | PCC: {p:.4f} | RMSD: {r:.4f} "
              f"| Train: {len(np.unique(groups[train_idx]))} pts "
              f"| Test: {len(np.unique(groups[test_idx]))} pts")

        if p > best_pcc:
            best_pcc = p
            best_fold_data = {
                'fold':          fold + 1,
                'member_models': [
                    {
                        'centers':           m.centers.detach().clone(),
                        'sigmas':            m.sigmas.detach().clone(),
                        'consequent_params': m.consequent_params.detach().clone(),
                    }
                    for m in member_models
                ],
                'scaler_x':      sc_x,
                'scaler_y':      sc_y,
                'feature_names': ELITE_5,
                'n_clusters':    N_CLUSTERS,
                'pcc':           best_pcc,
                'y_train_raw':   y_tr.flatten().tolist(),
            }

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(best_fold_data, SAVE_PATH)
    print(f"\nSaved best model (Fold {best_fold_data['fold']}, "
          f"PCC {best_pcc:.4f}) → {SAVE_PATH}")

    print("\n" + "=" * 45)
    print("      FINAL PUBLICATION STATISTICS")
    print("=" * 45)
    print(f"Mean PCC:  {np.mean(fold_pccs):.4f} (± {np.std(fold_pccs):.4f})")
    print(f"Mean RMSD: {np.mean(fold_rmsds):.4f} (± {np.std(fold_rmsds):.4f})")
    print("=" * 45)

    return fold_pccs, fold_rmsds


if __name__ == "__main__":
    run_final_validation()