# Scripts/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("Libraries loaded")

# Load dataset
df = pd.read_csv("PreProcessed-Data/alsfrs_demo_fvc_vitals_muscle.csv")


print("Dataset loaded")
print("Shape:", df.shape)

# ------------------------
# Drop identifier column
# ------------------------

df = df.drop(columns=["subject_id"])

# ------------------------
# Define target variable
# ------------------------

target = "ALSFRS_R_Total"

X = df.drop(columns=[target])
y = df[target]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ------------------------
# Train Test Split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# ------------------------
# Train Random Forest Model
# ------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training complete")

# ------------------------
# Predictions
# ------------------------

preds = model.predict(X_test)

# ------------------------
# Evaluation
# ------------------------

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)


import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Feature Importance
# ------------------------

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Feature Importance (Random Forest)")

plt.tight_layout()

plt.savefig("Analysis-Images/feature_importance.png")

plt.show()

print("\nFeature importance plot saved")