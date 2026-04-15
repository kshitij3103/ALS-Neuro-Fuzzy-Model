import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = 'analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    


try:
  
    df = pd.read_csv('data/Preprocessed-data/selector_model_tournament.csv')
except FileNotFoundError:
    print("Error: 'data/Preprocessed-data/selector_model_tournament.csv' nahi mili.")
    exit()

df['Combo'] = df['Selector'] + "\n(" + df['Model'] + ")"
top_5 = df.sort_values(by='Avg_PCC', ascending=False).head(5)


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(1, 2, figsize=(20, 9))

palette_pcc = sns.color_palette("viridis", n_colors=5)
sns.barplot(
    x='Avg_PCC', y='Combo', data=top_5, 
    ax=ax[0], palette=palette_pcc, hue='Combo', legend=False,
    edgecolor='black', linewidth=1.5
)

ax[0].set_title('Top 5 Models: Pearson Correlation (PCC) ↑', fontsize=22, fontweight='bold', pad=25)
ax[0].set_xlabel('Average PCC Score (Higher is Better)', fontsize=14, labelpad=15)
ax[0].set_ylabel('', fontsize=14)
ax[0].set_xlim(0.3, 0.45) 


for p in ax[0].patches:
    ax[0].annotate(f'{p.get_width():.4f}', 
                   (p.get_width() + 0.005, p.get_y() + p.get_height()/2), 
                   ha='left', va='center', fontsize=13, fontweight='bold', color='darkgreen')


palette_rmsd = sns.color_palette("magma_r", n_colors=5) 
sns.barplot(
    x='Avg_RMSD', y='Combo', data=top_5, 
    ax=ax[1], palette=palette_rmsd, hue='Combo', legend=False,
    edgecolor='black', linewidth=1.5
)

ax[1].set_title('Top 5 Models: Prediction Error (RMSD) ↓', fontsize=22, fontweight='bold', pad=25)
ax[1].set_xlabel('Average RMSD Score (Lower is Better)', fontsize=14, labelpad=15)
ax[1].set_ylabel('', fontsize=14)
ax[1].set_xlim(0.5, 0.6)


for p in ax[1].patches:
    ax[1].annotate(f'{p.get_width():.4f}', 
                   (p.get_width() + 0.005, p.get_y() + p.get_height()/2), 
                   ha='left', va='center', fontsize=13, fontweight='bold', color='darkred')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('ALS Progression Tournament: Performance Benchmarking', fontsize=28, fontweight='bold', y=0.98)


save_path = os.path.join(output_dir, 'model_performance_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n--- SUCCESS ---")
print(f"Graph aesthetic_model_comparison.png ab '{output_dir}' ")