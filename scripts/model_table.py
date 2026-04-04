import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Setup Data from your Benchmark Graph
data = {
    'Feature Selection': [
        'RF Importance', 
        'Pearson (f_regression)', 
        'Pearson (f_regression)', 
        'RF Importance', 
        'RF Importance'
    ],
    'Model': [
        'Random Forest', 
        'Random Forest', 
        'XGBoost', 
        'XGBoost', 
        'Linear Regression'
    ],
    'PCC (↑)': [0.4130, 0.4118, 0.4011, 0.3883, 0.3690],
    'RMSD (↓)': [0.5423, 0.5441, 0.5506, 0.5535, 0.5529]
}

df = pd.DataFrame(data)

# 2. Aesthetic Table Creation (Matplotlib)
def save_aesthetic_table(dataframe, filename):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.axis('tight')

    # Color Palette - Matching your previous graphs
    header_color = '#2c3e50'
    row_colors = ['#f8f9fa', '#ffffff']
    
    # Create Table
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc='center',
        loc='center'
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5) # Scale for better padding

    for (row, col), cell in table.get_celld().items():
        if row == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)
        else: # Rows
            cell.set_facecolor(row_colors[row % len(row_colors)])
        
        # Highlight the winning row (Winner of Tournament)
        if row == 1:
            cell.set_text_props(weight='bold', color='#1e8449') # Dark Green

    plt.title('ALS Progression Tournament: Model Performance Summary', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save Image
    output_path = f'analysis/{filename}.png'
    if not os.path.exists('analysis'): os.makedirs('analysis')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Table image saved at: {output_path}")

# 3. Print Formatted Output
print("\n" + "="*60)
print("       ALS PROGRESSION TOURNAMENT: SUMMARY RESULTS")
print("="*60)
print(df.to_string(index=False))
print("="*60)

# Run Visualization
save_aesthetic_table(df, 'tournament_summary_table')