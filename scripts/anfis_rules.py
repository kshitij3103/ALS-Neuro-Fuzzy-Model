import pandas as pd
import numpy as np

# 1. Data Load
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')

# Features & Target
elite_5 = ['Onset_Duration', 'ALSFRS_Total_Unified_median', 'FVC_slope', 'Weight_slope', 'Q1_Speech_min']
target = 'target_slope'

# 2. Categorizing Features (Safe Ranking to avoid duplicates error)
def get_bins_safe(df, column):
    return pd.qcut(df[column].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])

df_fuzzy = pd.DataFrame()
for col in elite_5:
    df_fuzzy[col] = get_bins_safe(data, col)

# 3. Categorizing Target
def categorize_target(s):
    if s < -1.1: return 'FAST'
    elif s < -0.5: return 'MEDIUM'
    else: return 'SLOW'

df_fuzzy['Outcome'] = data[target].apply(categorize_target)

# 4. Extracting Rules with Support
# Observed=False helps with categorical grouping
rules_summary = df_fuzzy.groupby(elite_5, observed=False)['Outcome'].agg([
    ('Rule_Outcome', lambda x: x.mode()[0] if not x.mode().empty else None),
    ('Support', 'count')
]).reset_index()

# Filter out empty combinations
rules_summary = rules_summary[rules_summary['Support'] > 0]

# 5. Extracting the 5-3-2 Mix with Highest Support
top_fast = rules_summary[rules_summary['Rule_Outcome'] == 'FAST'].sort_values('Support', ascending=False).head(2)
top_med = rules_summary[rules_summary['Rule_Outcome'] == 'MEDIUM'].sort_values('Support', ascending=False).head(5)
top_slow = rules_summary[rules_summary['Rule_Outcome'] == 'SLOW'].sort_values('Support', ascending=False).head(3)

# Combine and Shuffle
final_rules = pd.concat([top_fast, top_med, top_slow]).sample(frac=1).reset_index(drop=True)

# 6. Final Output (Fixed Indexing)
print(f"{'No.':<4} | {'Fuzzy Rule (Onset, ALSFRS, FVC, Weight, Speech)':<60} | {'Outcome':<8} | {'Support'}")
print("-" * 108)

for i, row in final_rules.iterrows():
    # .iloc use karke hum position-based access kar rahe hain
    # elite_5 are columns 0 to 4
    combo = f"IF ({row.iloc[0]}, {row.iloc[1]}, {row.iloc[2]}, {row.iloc[3]}, {row.iloc[4]})"
    print(f"{i+1:<4} | {combo:<60} | {row['Rule_Outcome']:<8} | {row['Support']} patients")

print("\n" + "="*50)
print(f"Unique Rule Combinations Found in Data: {len(rules_summary)}")
print("="*50)