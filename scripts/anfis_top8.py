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
DATA_PATH   = 'data/Preprocessed-data/MASTER_RAW_DATA.csv'
SAVE_PATH   = 'data/Preprocessed-data/trained_anfis_ensemble.pth'
ELITE_5     = ['Onset_Duration', 'ALSFRS_Total_Unified_median',
               'FVC_slope', 'Weight_slope', 'Q1_Speech_min']
TARGET_COL  = 'target_slope'
SUBJECT_COL = 'subject_id'       # <-- change if your column is named differently
N_FOLDS     = 5
N_EPOCHS    = 20
N_CLUSTERS  = 10
ENSEMBLE_SEEDS = [42, 52, 62]


# ==========================================
# HYBRID ANFIS ARCHITECTURE
# ==========================================
class ClusterANFIS(nn.Module):
    def __init__(self, num_inputs=5, num_clusters=10, initial_centers=None):
        super(ClusterANFIS, self).__init__()
        self.num_inputs   = num_inputs
        self.num_clusters = num_clusters
        # Premise parameters — updated by Adam
        self.centers = nn.Parameter(initial_centers.clone())
        self.sigmas  = nn.Parameter(torch.ones(num_clusters, num_inputs) * 1.5)
        # Consequent parameters — updated by LSE (not nn.Parameter intentionally)
        self.register_buffer(
            'consequent_params',
            torch.zeros(num_clusters * (num_inputs + 1), 1)
        )

    def forward_premise(self, x):
        """Returns normalised firing strengths  [batch, clusters]."""
        x_expanded = x.unsqueeze(1)                                   # [B, 1, F]
        s = torch.nn.functional.softplus(self.sigmas) + 1e-5          # always > 0
        mu = torch.exp(-((x_expanded - self.centers) ** 2) / (2 * s ** 2))
        w  = torch.prod(mu, dim=2)                                    # [B, C]
        return w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)

    def construct_design_matrix(self, x, w_norm):
        """Builds the TSK consequent design matrix  [batch, C*(F+1)]."""
        batch_size = x.shape[0]
        x_ext = torch.cat([x, torch.ones(batch_size, 1)], dim=1)     # [B, F+1]
        return (w_norm.unsqueeze(2) * x_ext.unsqueeze(1)).view(batch_size, -1)

    def forward(self, x):
        w_norm = self.forward_premise(x)
        A      = self.construct_design_matrix(x, w_norm)
        return torch.matmul(A, self.consequent_params)

    def lse_update(self, x, y, lam=2.0):
        """
        Least-squares update for consequent parameters (hybrid learning).
        Called BEFORE the Adam gradient step each epoch.
        """
        with torch.no_grad():
            w_norm = self.forward_premise(x)
            A      = self.construct_design_matrix(x, w_norm)
            I      = torch.eye(A.shape[1], device=A.device)
            AtA    = torch.matmul(A.T, A) + lam * I
            new_params = torch.linalg.solve(AtA, torch.matmul(A.T, y))
            self.consequent_params.copy_(new_params)


# ==========================================
# FOLD-LEVEL IMPUTATION  (FIX 1 — no leakage)
# ==========================================
def impute_with_train_median(X_tr: np.ndarray,
                              X_te: np.ndarray):
    """Compute median on train only, apply to both splits."""
    medians = np.nanmedian(X_tr, axis=0)
    for col in range(X_tr.shape[1]):
        X_tr[:, col] = np.where(np.isnan(X_tr[:, col]), medians[col], X_tr[:, col])
        X_te[:, col] = np.where(np.isnan(X_te[:, col]), medians[col], X_te[:, col])
    return X_tr, X_te


# ==========================================
# SINGLE ENSEMBLE MEMBER TRAINING
# ==========================================
def train_member(X_tr_scaled: torch.Tensor,
                 y_tr_scaled: torch.Tensor,
                 seed: int,
                 n_epochs: int = N_EPOCHS,
                 n_clusters: int = N_CLUSTERS) -> ClusterANFIS:
    """Train one ANFIS member with a given KMeans seed."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X_tr_scaled.numpy())
    init_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    model = ClusterANFIS(num_inputs=X_tr_scaled.shape[1],
                         num_clusters=n_clusters,
                         initial_centers=init_centers)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(n_epochs):
        # --- Consequent update (LSE) ---
        model.lse_update(X_tr_scaled, y_tr_scaled)

        # --- Premise update (gradient) ---
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
def run_final_validation(k: int = N_FOLDS):
    # --- Load data (NaNs preserved — imputation happens inside fold) ---
    data = pd.read_csv(DATA_PATH)

    missing = [c for c in ELITE_5 + [TARGET_COL, SUBJECT_COL] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X      = data[ELITE_5].values.astype(float)      # NaNs kept intentionally
    y      = data[TARGET_COL].values.reshape(-1, 1)
    groups = data[SUBJECT_COL].values                 # FIX 2 — group by patient

    # GroupKFold — same patient never spans train and test
    gkf = GroupKFold(n_splits=k)

    fold_pccs, fold_rmsds = [], []
    best_pcc        = -np.inf
    best_fold_data  = {}

    print(f"\nStarting {k}-Fold Validation "
          f"(GroupKFold | Ensemble {len(ENSEMBLE_SEEDS)} members | {N_EPOCHS} Epochs)...")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr_raw, X_te_raw = X[train_idx].copy(), X[test_idx].copy()
        y_tr, y_te         = y[train_idx],          y[test_idx]

        # FIX 1 — impute using train statistics only
        X_tr_raw, X_te_raw = impute_with_train_median(X_tr_raw, X_te_raw)

        # Scale (fit on train, transform both)
        sc_x, sc_y = StandardScaler(), StandardScaler()
        X_tr_scaled = torch.tensor(sc_x.fit_transform(X_tr_raw), dtype=torch.float32)
        y_tr_scaled = torch.tensor(sc_y.fit_transform(y_tr),     dtype=torch.float32)
        X_te_scaled = torch.tensor(sc_x.transform(X_te_raw),     dtype=torch.float32)

        # Train ensemble members and collect predictions
        member_models = []
        member_preds  = []

        for seed in ENSEMBLE_SEEDS:
            model = train_member(X_tr_scaled, y_tr_scaled, seed=seed)
            member_models.append(model)
            with torch.no_grad():
                pred_scaled = model(X_te_scaled).numpy()
            member_preds.append(pred_scaled)

        # Ensemble: average predictions then inverse-transform
        ensemble_pred_scaled = np.mean(member_preds, axis=0)
        y_pred = sc_y.inverse_transform(ensemble_pred_scaled)

        p, _ = pearsonr(y_te.flatten(), y_pred.flatten())
        r    = np.sqrt(np.mean((y_te - y_pred) ** 2))

        fold_pccs.append(p)
        fold_rmsds.append(r)
        print(f"Fold {fold+1} Completed | PCC: {p:.4f} | RMSD: {r:.4f} "
              f"| Train patients: {len(np.unique(groups[train_idx]))} "
              f"| Test patients: {len(np.unique(groups[test_idx]))}")

        # FIX 3 — save ALL ensemble members, not just last
        if p > best_pcc:
            best_pcc = p
            best_fold_data = {
                'fold':             fold + 1,
                'member_models':    [
                    {
                        'centers':            m.centers.detach().clone(),
                        'sigmas':             m.sigmas.detach().clone(),
                        'consequent_params':  m.consequent_params.detach().clone(),
                    }
                    for m in member_models
                ],
                'scaler_x':         sc_x,
                'scaler_y':         sc_y,
                'feature_names':    ELITE_5,
                'pcc':              p,
            }

    # --- Save best fold ---
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(best_fold_data, SAVE_PATH)
    print(f"\nSaved Best Model from Fold {best_fold_data['fold']} "
          f"(PCC: {best_pcc:.4f}) to {SAVE_PATH}")

    # --- Final stats ---
    print("\n" + "=" * 45)
    print("      FINAL PUBLICATION STATISTICS")
    print("=" * 45)
    print(f"Mean PCC:  {np.mean(fold_pccs):.4f} (± {np.std(fold_pccs):.4f})")
    print(f"Mean RMSD: {np.mean(fold_rmsds):.4f} (± {np.std(fold_rmsds):.4f})")
    print("=" * 45)

    # Warn if results look suspiciously high (possible grouping issue)
    if np.mean(fold_pccs) > 0.65:
        print("\n[WARNING] PCC seems high — double-check that SUBJECT_COL "
              "is the correct patient identifier column.")

    return fold_pccs, fold_rmsds


if __name__ == "__main__":
    run_final_validation()