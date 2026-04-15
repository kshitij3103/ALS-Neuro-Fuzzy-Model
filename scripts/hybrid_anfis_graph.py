import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIG
# ==========================================
DATA_PATH      = 'data/Preprocessed-data/MASTER_RAW_DATA.csv'
ELITE_5        = ['Onset_Duration', 'ALSFRS_Total_Unified_median',
                  'FVC_slope', 'Weight_slope', 'Q1_Speech_min']
TARGET_COL     = 'target_slope'
SUBJECT_COL    = 'subject_id'
N_EPOCHS       = 20
N_CLUSTERS     = 10
LSE_LAMBDA     = 0.1
ENSEMBLE_SEEDS = [42, 52, 62]
PLOT_FOLD      = 1


class ClusterANFIS(nn.Module):
    def __init__(self, num_inputs=5, num_clusters=10, initial_centers=None):
        super().__init__()
        self.num_inputs   = num_inputs
        self.num_clusters = num_clusters
        self.centers = nn.Parameter(initial_centers.clone())
        self.sigmas  = nn.Parameter(torch.ones(num_clusters, num_inputs) * 1.5)
        self.register_buffer('consequent_params',
                             torch.zeros(num_clusters * (num_inputs + 1), 1))

    def forward_premise(self, x):
        s  = torch.nn.functional.softplus(self.sigmas) + 1e-5
        mu = torch.exp(-((x.unsqueeze(1) - self.centers) ** 2) / (2 * s ** 2))
        w  = torch.prod(mu, dim=2)
        return w / (w.sum(dim=1, keepdim=True) + 1e-8)

    def construct_design_matrix(self, x, w_norm):
        x_ext = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        return (w_norm.unsqueeze(2) * x_ext.unsqueeze(1)).view(x.shape[0], -1)

    def forward(self, x):
        w = self.forward_premise(x)
        return torch.matmul(self.construct_design_matrix(x, w), self.consequent_params)

    def lse_update(self, x, y):
        with torch.no_grad():
            w   = self.forward_premise(x)
            A   = self.construct_design_matrix(x, w)
            I   = torch.eye(A.shape[1])
            self.consequent_params.copy_(
                torch.linalg.solve(A.T @ A + LSE_LAMBDA * I, A.T @ y)
            )


def impute(X_tr, X_te):
    meds = np.nanmedian(X_tr, axis=0)
    for c in range(X_tr.shape[1]):
        X_tr[:, c] = np.where(np.isnan(X_tr[:, c]), meds[c], X_tr[:, c])
        X_te[:, c] = np.where(np.isnan(X_te[:, c]), meds[c], X_te[:, c])
    return X_tr, X_te


def generate_graph():
    data   = pd.read_csv(DATA_PATH)
    X      = data[ELITE_5].values.astype(float)
    y      = data[TARGET_COL].values.reshape(-1, 1)
    groups = data[SUBJECT_COL].values

    splits = list(GroupKFold(n_splits=5).split(X, y, groups))
    train_idx, test_idx = splits[PLOT_FOLD - 1]

    X_tr, X_te = impute(X[train_idx].copy(), X[test_idx].copy())
    y_tr, y_te = y[train_idx], y[test_idx]

    sc_x, sc_y  = StandardScaler(), StandardScaler()
    Xtr = torch.tensor(sc_x.fit_transform(X_tr), dtype=torch.float32)
    ytr = torch.tensor(sc_y.fit_transform(y_tr), dtype=torch.float32)
    Xte = torch.tensor(sc_x.transform(X_te),     dtype=torch.float32)

    train_losses = np.zeros(N_EPOCHS)
    test_pccs    = np.zeros(N_EPOCHS)

    for seed in ENSEMBLE_SEEDS:
        km  = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=10).fit(Xtr.numpy())
        mdl = ClusterANFIS(5, N_CLUSTERS,
                           torch.tensor(km.cluster_centers_, dtype=torch.float32))
        opt = torch.optim.Adam(mdl.parameters(), lr=0.01)

        for ep in range(N_EPOCHS):
            mdl.lse_update(Xtr, ytr)
            mdl.train()
            opt.zero_grad()
            loss = nn.MSELoss()(mdl(Xtr), ytr)
            loss.backward()
            opt.step()
            train_losses[ep] += loss.item()

            mdl.eval()
            with torch.no_grad():
                pred = sc_y.inverse_transform(mdl(Xte).numpy())
            p, _ = pearsonr(y_te.flatten(), pred.flatten())
            test_pccs[ep] += p

    train_losses /= len(ENSEMBLE_SEEDS)
    test_pccs    /= len(ENSEMBLE_SEEDS)
    epochs        = list(range(1, N_EPOCHS + 1))
    best_ep       = int(np.argmax(test_pccs)) + 1

    print(f"Best PCC: epoch {best_ep} → {test_pccs[best_ep-1]:.4f}")
    print(f"Epoch 20 PCC: {test_pccs[-1]:.4f}")

    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#FAFAFA')

    # Training loss
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Training Loss (MSE)', color='#C0392B', fontsize=11)
    ax1.plot(epochs, train_losses, color='#C0392B', linewidth=2.5,
             label='Training Loss (MSE)')
    ax1.tick_params(axis='y', labelcolor='#C0392B')
    ax1.set_xlim(1, N_EPOCHS)

    # Validation PCC
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation PCC', color='#2471A3', fontsize=11)
    ax2.plot(epochs, test_pccs, color='#2471A3', linewidth=2.5,
             label='Validation PCC')
    ax2.tick_params(axis='y', labelcolor='#2471A3')

    
    ax2.axvline(x=best_ep, color='#27AE60', linestyle='--',
                linewidth=1.8, label=f'Best PCC (Epoch {best_ep})')

    
    if best_ep != 20:
        ax2.axvline(x=20, color='#8E44AD', linestyle=':',
                    linewidth=1.8, label='Selected Cutoff (Epoch 20)')

    
    ax2.text(best_ep + 0.3, ax2.get_ylim()[1] * 0.995,
             f'Peak (Ep {best_ep})', color='#27AE60',
             fontsize=8, va='top')

    plt.title('ANFIS Training Analysis: Validation PCC vs. Training Loss\n'
              f'(Fold {PLOT_FOLD} | GroupKFold | {N_CLUSTERS} Clusters | λ={LSE_LAMBDA})',
              fontsize=12, fontweight='bold', pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower left', fontsize=9, framealpha=0.9)

    ax1.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()

    out = 'data/Preprocessed-data/Training_Analysis_Graph.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    try:
        generate_graph()
    except Exception as e:
        import traceback
        traceback.print_exc()