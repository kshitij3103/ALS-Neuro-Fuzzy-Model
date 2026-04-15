import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import os

output_dir = 'analysis'
os.makedirs(output_dir, exist_ok=True)


data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')

X = data.drop(columns=['subject_id', 'target_slope'])
y = data['target_slope']


X_train_raw, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)


target_feature = 'Creatinine_median'
train_series = X_train_raw[[target_feature]].copy()
original_non_nulls = train_series.dropna()

print(f"Analyzing Imputation Impact on: {target_feature}")


mean_imp = SimpleImputer(strategy='mean')
data_mean = mean_imp.fit_transform(train_series).flatten()

knn_imp = KNNImputer(n_neighbors=5)
data_knn = knn_imp.fit_transform(train_series).flatten()

original_data = original_non_nulls[target_feature]


sns.set_theme(style="whitegrid", font_scale=1.3)
plt.figure(figsize=(16, 9))


sns.kdeplot(data_knn, 
            label='KNN Imputation', 
            color='green', 
            linewidth=3)

sns.kdeplot(data_mean, 
            label='Mean Imputation', 
            color='red', 
            linewidth=3, 
            linestyle='--')

sns.kdeplot(original_data, 
            label='Original Data (Clean)', 
            color='black', 
            linewidth=3.5, 
            linestyle='-.')


plt.title('Data Quality Check: Mean vs. KNN Imputation', 
          fontsize=24, fontweight='bold', pad=20)

plt.suptitle(f'Feature: {target_feature} (80% Training Split)', 
             fontsize=14, y=0.92)

plt.xlabel(f'{target_feature}', fontsize=16)
plt.ylabel('Density', fontsize=16)


plt.legend(fontsize=13, loc='upper right', frameon=True)


mean_val = original_data.mean()

plt.annotate(
    'Mean and KNN imputation produce\nvery similar distributions',
    xy=(mean_val, 0.02),
    xytext=(mean_val + 25, 0.028),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=13,
    color='black',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black")
)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])

save_path = os.path.join(output_dir, 'imputation_comparison_fixed.png')
plt.savefig(save_path, dpi=300)

plt.show()

print(f"✅ Graph saved at: {save_path}")