import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup
output_dir = 'analysis'
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')

# 2. Category Create Karein (Based on your previous plot threshold -1.1)
threshold = -1.1
data['Progression_Category'] = ['Fast' if x < threshold else 'Slow' for x in data['target_slope']]

# 3. Select Top 3 Features from your RF Importance
top_3 = ['Onset_Duration', 'ALSFRS_Total_Unified_first', 'FVC_slope']

# 4. AESTHETIC VISUALIZATION
sns.set_theme(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for i, feat in enumerate(top_3):
    sns.boxplot(x='Progression_Category', y=feat, data=data, ax=axes[i], 
                palette={'Fast': '#F39C12', 'Slow': '#2E86C1'}, hue='Progression_Category', legend=False)
    axes[i].set_title(f'{feat} vs Progression', fontsize=15, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Clinical Value')

plt.suptitle('Clinical Validation: Top Features across Progression Groups', fontsize=22, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clinical_validation_boxplots.png'), dpi=300)
plt.show()