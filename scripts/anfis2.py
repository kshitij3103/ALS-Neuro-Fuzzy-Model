import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# 1. CLEAN DATA LOAD
elite_5 = ['Onset_Duration', 'ALSFRS_Total_Unified_median', 'FVC_slope', 'Weight_slope', 'Q1_Speech_min']
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')

X = data[elite_5].fillna(data[elite_5].median()).values
y = data['target_slope'].values.reshape(-1, 1)

# Using StandardScaler for better stability
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    scaler_x.fit_transform(X), 
    scaler_y.fit_transform(y), 
    test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# 2. STABLE NEURO-FUZZY MODEL
class StableANFIS(nn.Module):
    def __init__(self):
        super(StableANFIS, self).__init__()
        # Bottleneck architecture to force pattern recognition
        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.Tanh(), # Tanh is better for normalized data (-1 to 1)
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

# 3. FINE-TUNED TRAINING
model = StableANFIS()
# Lower Learning Rate (0.001) for more stable convergence
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

print("Training Stable Elite-5 Model...")
for epoch in range(400): # More epochs for slow learning
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# 4. RESULTS
model.eval()
with torch.no_grad():
    y_pred_std = model(X_test_t).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_std)
    y_actual = scaler_y.inverse_transform(y_test)

pcc, _ = pearsonr(y_actual.flatten(), y_pred.flatten())
rmsd = np.sqrt(np.mean((y_actual - y_pred)**2))

print("\n" + "="*40)
print(f"   RE-ALIGNED ANFIS RESULTS")
print("="*40)
print(f"PCC Score:  {pcc:.4f}")
print(f"RMSD Error: {rmsd:.4f}")
print("="*40)