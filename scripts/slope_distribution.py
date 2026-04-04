import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup
output_dir = 'analysis'
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 2. Load Data
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')
slope_data = data['target_slope']

# 3. Define Threshold (Standard ALS threshold is -1.1)
threshold = -1.1
fast_mask = slope_data < threshold

# 4. AESTHETIC VISUALIZATION
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 8))

# Histogram with two colors
# Fast Progressors (Orange)
sns.histplot(slope_data[fast_mask], bins=25, color='#F39C12', label='Fast Progression', alpha=0.9, edgecolor='black')

# Medium-Slow Progressors (Blue)
sns.histplot(slope_data[~fast_mask], bins=25, color='#2E86C1', label='Medium-Slow Progression', alpha=0.9, edgecolor='black')

# Threshold line
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

# Labels & Aesthetics
plt.title('Figure 2: Distribution of ALSFRS Slope and Progression Categories', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Slope (ALSFRS points/month)', fontsize=14, fontweight='bold')
plt.ylabel('Patient Count', fontsize=14, fontweight='bold')
plt.legend(frameon=True, shadow=True, fontsize=12)

# Text Annotations as per Image
plt.text(-2.5, 200, 'Fast\nProgression', color='#D35400', fontweight='bold', ha='center', fontsize=13)
plt.text(0.5, 200, 'Medium-Slow\nProgression', color='#1B4F72', fontweight='bold', ha='center', fontsize=13)

# Grid and Spacing
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Save to Analysis Folder
save_path = os.path.join(output_dir, 'target_slope_distribution.png')
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Success! Figure 2 saved in {save_path}")