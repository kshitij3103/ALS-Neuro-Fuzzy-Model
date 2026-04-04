import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Setup & Data Loading
if not os.path.exists('analysis'): os.makedirs('analysis')
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')

elite_5 = ['Onset_Duration', 'ALSFRS_Total_Unified_median', 'FVC_slope', 'Weight_slope', 'Q1_Speech_min']
target = 'target_slope'

# 2. Get Raw Boundaries (Terciles)
def get_bounds(df, col):
    # Using actual quantiles for the legend and labels
    q33, q66 = df[col].quantile(0.33), df[col].quantile(0.66)
    return {
        'Low': f"<{q33:.1f}", 
        'Medium': f"{q33:.1f}-{q66:.1f}", 
        'High': f">{q66:.1f}"
    }

boundary_map = {col: get_bounds(data, col) for col in elite_5}

# 3. Categorize Features & Outcome
def get_bins_safe(df, col):
    # Safe ranking ensures we get 3 bins even with duplicate values
    return pd.qcut(df[col].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])

df_fuzzy = pd.DataFrame({col: get_bins_safe(data, col) for col in elite_5})
df_fuzzy['Outcome'] = data[target].apply(lambda s: 'FAST' if s < -1.1 else ('MEDIUM' if s < -0.5 else 'SLOW'))

# 4. Group by Support (Finding Top 10 Global Trends)
rules = df_fuzzy.groupby(elite_5, observed=False)['Outcome'].agg([
    ('Outcome', lambda x: x.mode()[0] if not x.mode().empty else None),
    ('Support', 'count')
]).reset_index().dropna().sort_values('Support', ascending=False).head(10)

# 5. Prepare Data for the Table
table_data = []
for i, (_, row) in enumerate(rules.iterrows()):
    row_list = [i+1] # Rank
    for col in elite_5:
        label = row[col]
        raw_range = boundary_map[col][label]
        # Format: "Medium\n(500-1200)"
        row_list.append(f"{label}\n({raw_range})")
    row_list.append(row['Outcome'])
    row_list.append(row['Support'])
    table_data.append(row_list)

columns = ['Rank', 'Onset\nDuration', 'ALSFRS\nMedian', 'FVC\nSlope', 'Weight\nSlope', 'Speech\n(Q1)', 'Outcome', 'Support']

# 6. Visualization with Matplotlib
fig, ax = plt.subplots(figsize=(18, 12))
ax.axis('off')

# Color Mapping for Outcomes
colors = {'FAST': '#ffadad', 'MEDIUM': '#ffd6a5', 'SLOW': '#caffbf'}
cell_colors = []

for row_idx in range(len(table_data)):
    # Row colors: White for features, category-specific for Outcome
    current_row_colors = ['#ffffff'] * 6 # Rank + 5 Features
    current_row_colors.append(colors[table_data[row_idx][-2]]) # Outcome Column
    current_row_colors.append('#ffffff') # Support Column
    cell_colors.append(current_row_colors)

# Create Table
plt_table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    cellColours=cell_colors
)

# Styling
plt_table.auto_set_font_size(False)
plt_table.set_fontsize(11)
plt_table.scale(1.1, 4.2) # High scaling to fit newlines

# Bold Headers
for (row, col), cell in plt_table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_facecolor('#2c3e50') # Professional dark blue header

plt.title('Table 4: Top 10 Data-Driven Fuzzy Rules by Patient Support (ALS Study)', 
          fontsize=20, fontweight='bold', pad=40)

# Save
save_path = 'analysis/top_10_rules_fixed.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Table Saved: {save_path}")