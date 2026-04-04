import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Setup Data
if not os.path.exists('analysis'): os.makedirs('analysis')

comparison_data = [
    # Category | Model | PCC | RMSD
    ['Baseline', 'Linear Regression', 0.3690, 0.5529],
    ['Baseline', 'Random Forest', 0.4130, 0.5423],
    ['Proposed', 'ANFIS (Neuro-Fuzzy)', 0.3977, 0.5292],
    
    ['Literature', 'CNN', 0.4390, 0.5270],
    ['Literature', 'FFNN', 0.4510, 0.5280],
    ['Literature', 'FFNN + CNN', 0.4620, 0.5210]
]

df_comp = pd.DataFrame(comparison_data, columns=['Category', 'Model/Methodology', 'PCC (↑)', 'RMSD (↓)'])

# 2. Plotting
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Table Colors
header_color = '#2c3e50'
row_colors = []
for cat in df_comp['Category']:
    if cat == 'Proposed': row_colors.append(['#e8f8f5', '#e8f8f5', '#e8f8f5', '#e8f8f5']) # Highlight Green
    elif cat == 'Baseline': row_colors.append(['#fdf2e9', '#fdf2e9', '#fdf2e9', '#fdf2e9']) # Light Orange
    else: row_colors.append(['#ffffff', '#ffffff', '#ffffff', '#ffffff']) # White

# Create Table
tab = ax.table(
    cellText=df_comp.values,
    colLabels=df_comp.columns,
    cellLoc='center',
    loc='center',
    cellColours=row_colors
)

# Styling
tab.auto_set_font_size(False)
tab.set_fontsize(12)
tab.scale(1.2, 2.5)

# Bold Headers & Proposed Row
for (row, col), cell in tab.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor(header_color)
    if row == 3: # ANFIS Row
        cell.set_text_props(weight='bold')

plt.title('Table 5: Comparative Performance Analysis\n(Proposed ANFIS vs Baselines vs Literature)', 
          fontsize=16, fontweight='bold', pad=20)

# Save
output_path = 'analysis/final_model_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Tale Saved {output_path}")