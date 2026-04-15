import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


output_dir = 'analysis'
if not os.path.exists(output_dir): os.makedirs(output_dir)


data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')


top_10_plus_target = [
    'Onset_Duration', 'ALSFRS_Total_Unified_first', 'ALSFRS_Total_Unified_median',
    'ALSFRS_Total_Unified_last', 'ALSFRS_Total_Unified_slope', 'ALSFRS_Total_Unified_min',
    'ALSFRS_Total_Unified_max', 'Weight_slope', 'FVC_slope', 'Q1_Speech_min',
    'target_slope' 
]

corr_matrix = data[top_10_plus_target].corr()

plt.figure(figsize=(15, 12))
sns.set_theme(style="white")




heatmap = sns.heatmap(
    corr_matrix, 
    
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', #red +ve blue -ve
    vmin=-1, vmax=1, 
    center=0,
    square=True, 
    linewidths=1.2, 
    cbar_kws={"shrink": .7},
    annot_kws={"size": 10, "weight": "bold"}
)

plt.title('Top 10 Clinical Features: Correlation Matrix (PCC)', fontsize=24, fontweight='bold', pad=30)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
save_path = os.path.join(output_dir, 'heatmap.png')
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Success! Correlation Heatmap saved in {save_path}")