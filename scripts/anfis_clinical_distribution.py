import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


output_dir = 'analysis'
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')
slope = data['target_slope']

#  Logic for 3 Categories
def categorize(s):
    if s < -1: return 'Fast'
    elif s < -0.5: return 'Medium'
    else: return 'Slow'

data['Category'] = data['target_slope'].apply(categorize)


plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# Colors: Red (Fast), Yellow (Medium), Green (Slow) - Universal Clinical Colors
colors = {'Fast': '#E74C3C', 'Medium': '#F1C40F', 'Slow': '#27AE60'}

sns.histplot(data=data, x='target_slope', hue='Category', palette=colors, 
             multiple="stack", bins=40, edgecolor='black', alpha=0.8)


plt.axvline(-1, color='black', linestyle='--', alpha=0.6)
plt.axvline(-0.5, color='black', linestyle='--', alpha=0.6)

plt.title('Figure 2 (Updated): 3-Tier ALS Progression Distribution', fontsize=20, fontweight='bold')
plt.xlabel('ALSFRS Slope (points/month)', fontsize=14)
plt.ylabel('Patient Count', fontsize=14)


plt.text(-2.5, 100, 'FAST', color='#C0392B', fontweight='bold', fontsize=14)
plt.text(-0.8, 150, 'MEDIUM', color='#9A7D0A', fontweight='bold', fontsize=14)
plt.text(0.2, 200, 'SLOW', color='#1E8449', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'three_category_distribution.png'), dpi=300)
plt.show()