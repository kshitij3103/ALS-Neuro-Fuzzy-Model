import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import os

# 1. Directory Setup
output_dir = 'analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Load Raw Data
print("Loading Raw Data...")
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')
X = data.drop(columns=['subject_id', 'target_slope'])
y = data['target_slope']

# 3. STEP 1: 80/20 Split (Zero Leakage Rule)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. STEP 2: KNN Imputation (Training Set Only)
print("Performing KNN Imputation (n=5) on Training Set...")
# KNN handle missing values by looking at similar patients
knn_imputer = KNNImputer(n_neighbors=5)
X_train_imputed = knn_imputer.fit_transform(X_train_raw)

# 5. STEP 3: Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X.columns)

print("Calculating Feature Rankings (4 Selectors)...")

# --- A. Pearson (Linear) ---
f_selector = SelectKBest(score_func=f_regression, k='all').fit(X_train_scaled, y_train)
pearson_scores = pd.DataFrame({'Feature': X.columns, 'Score': f_selector.scores_}).sort_values(by='Score', ascending=False)

# --- B. Mutual Information (Non-linear) ---
mi_selector = SelectKBest(score_func=mutual_info_regression, k='all').fit(X_train_scaled, y_train)
mi_scores = pd.DataFrame({'Feature': X.columns, 'Score': mi_selector.scores_}).sort_values(by='Score', ascending=False)

# --- C. Random Forest (Winner Model) ---
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_scores = pd.DataFrame({'Feature': X.columns, 'Score': rf.feature_importances_}).sort_values(by='Score', ascending=False)

# --- D. RFE (Recursive Elimination) ---
rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
rfe.fit(X_train_scaled, y_train)
rfe_scores = pd.DataFrame({'Feature': X.columns, 'Rank': rfe.ranking_})
rfe_scores['Score'] = 1 / rfe_scores['Rank'] 
rfe_scores = rfe_scores.sort_values(by='Score', ascending=False)

# 6. AESTHETIC VISUALIZATION
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(2, 2, figsize=(24, 18))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

selectors_data = [
    (pearson_scores, "Pearson Correlation", "magma", axes[0,0]),
    (mi_scores, "Mutual Information", "viridis", axes[0,1]),
    (rf_scores, "Random Forest Importance", "rocket", axes[1,0]),
    (rfe_scores, "RFE Ranking (Inverted)", "mako", axes[1,1])
]

for df_rank, title, color, ax in selectors_data:
    sns.barplot(x='Score', y='Feature', data=df_rank.head(20), ax=ax, palette=color, hue='Feature', legend=False, edgecolor='0.2')
    ax.set_title(f"Top 20: {title}", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Importance Metric", fontsize=12)
    ax.set_ylabel("")

plt.suptitle("Leakage-Free Feature Importance Comparison (KNN-Imputed Training Set)", fontsize=28, fontweight='bold', y=0.96)

# Save Plot
save_path = os.path.join(output_dir, 'feature_importance_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n--- SUCCESS ---")
print(f"Graph saved: {save_path}")