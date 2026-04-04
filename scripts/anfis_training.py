import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

# ==============================
# 1. DATA PREPARATION
# ==============================
print("Loading Data...")

data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')

elite_5 = [
    'Onset_Duration',
    'ALSFRS_Total_Unified_median',
    'FVC_slope',
    'Weight_slope',
    'Q1_Speech_min'
]

X = data[elite_5].values
y = data['target_slope'].values.reshape(-1, 1)

# Handle missing values
X = SimpleImputer(strategy='median').fit_transform(X)

# Scaling (important for ANFIS)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# ==============================
# 2. HYBRID ANFIS MODEL
# ==============================
class HybridANFIS(nn.Module):
    def __init__(self, n_inputs=5, n_mfs=3):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs  # 3^5 = 243
        
        # Premise parameters (GD)
        self.mu = nn.Parameter(torch.rand(n_inputs, n_mfs))
        self.sigma = nn.Parameter(torch.ones(n_inputs, n_mfs) * 0.5)
        
        # Consequents (LSE)
        self.consequents = None

    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def forward(self, x):
        B = x.shape[0]

        # Layer 1: Fuzzification
        x_exp = x.unsqueeze(2)
        mu = self.mu.unsqueeze(0)
        sigma = self.sigma.unsqueeze(0)
        
        mfs = self.gaussian(x_exp, mu, sigma)  # [B, inputs, mfs]

        # Layer 2: Rule firing (Cartesian product)
        rules = mfs[:, 0, :]
        for i in range(1, self.n_inputs):
            rules = rules.unsqueeze(2) * mfs[:, i, :].unsqueeze(1)
            rules = rules.reshape(B, -1)

        # Layer 3: Normalization
        norm_w = rules / (rules.sum(dim=1, keepdim=True) + 1e-8)

        return norm_w

    def compute_lse(self, x, y, norm_w):
        B = x.shape[0]

        # Add bias
        x_aug = torch.cat([x, torch.ones(B, 1)], dim=1)  # [B,6]

        A = []
        for i in range(self.n_rules):
            w = norm_w[:, i].unsqueeze(1)
            A.append(w * x_aug)

        A = torch.cat(A, dim=1)  # [B, 243*6]

        # Solve Least Squares
        theta = torch.linalg.lstsq(A, y).solution
        self.consequents = theta

    def predict(self, x):
        norm_w = self.forward(x)
        B = x.shape[0]

        x_aug = torch.cat([x, torch.ones(B, 1)], dim=1)

        outputs = []
        idx = 0

        for i in range(self.n_rules):
            params = self.consequents[idx:idx+6]
            f = x_aug @ params
            outputs.append(norm_w[:, i:i+1] * f)
            idx += 6

        return torch.sum(torch.stack(outputs), dim=0)


# ==============================
# 3. TRAINING (HYBRID)
# ==============================
print("Training Hybrid ANFIS...")

model = HybridANFIS()
optimizer = torch.optim.Adam([model.mu, model.sigma], lr=0.01)
criterion = nn.MSELoss()

for epoch in range(40):

    # Forward
    norm_w = model(X_train_t)

    # LSE step
    model.compute_lse(X_train_t, y_train_t, norm_w)

    # Prediction
    y_pred = model.predict(X_train_t)

    loss = criterion(y_pred, y_train_t)

    # Backward (only premise params)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")


# ==============================
# 4. EVALUATION
# ==============================
print("\nEvaluating Model...")

model.eval()
with torch.no_grad():
    y_pred_scaled = model.predict(X_test_t).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

pcc, _ = pearsonr(y_actual.flatten(), y_pred.flatten())
rmsd = np.sqrt(np.mean((y_actual - y_pred)**2))

print("\n" + "="*40)
print("   HYBRID ANFIS RESULTS")
print("="*40)
print(f"PCC Score:  {pcc:.4f}")
print(f"RMSD Error: {rmsd:.4f}")
print("="*40)