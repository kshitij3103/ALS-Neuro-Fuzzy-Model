# scripts/distribution_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries loaded")

# Load cleaned dataset
# Load cleaned dataset
df = pd.read_csv("PreProcessed-Data/alsfrs_cleaned.csv")

print("Dataset loaded")
print("Shape:", df.shape)

# -------------------------
# Target variable distribution
# -------------------------

plt.figure(figsize=(6,4))

sns.histplot(df["ALSFRS_R_Total"], bins=20, kde=True)

plt.title("ALSFRS-R Score Distribution")

plt.xlabel("ALSFRS-R Score")

plt.ylabel("Frequency")

plt.savefig("Analysis-Images/alsfrs_distribution.png")

plt.show()

print("Distribution plot saved")