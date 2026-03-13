# Scripts/heatmap_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Libraries loaded")

# Load processed dataset
df = pd.read_csv("PreProcessed-Data/alsfrs_demo_fvc_vitals_muscle.csv")
# drop identifier column
df = df.drop(columns=["subject_id"])

print("Dataset loaded")
print("Shape:", df.shape)

# ----------------------
# Correlation matrix
# ----------------------

corr = df.corr()
plt.figure(figsize=(14,10))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.subplots_adjust(bottom=0.50)
plt.tight_layout()

plt.savefig("Analysis-Images/correlation_heatmap.png")

plt.show()

print("Heatmap saved")