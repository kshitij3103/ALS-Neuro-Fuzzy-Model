import pandas as pd

# 1. Data Load
data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')

# Features & Target
elite_5 = ['Onset_Duration', 'ALSFRS_Total_Unified_median', 'FVC_slope', 'Weight_slope', 'Q1_Speech_min']
target = 'target_slope'

# 2. Categorizing Features (Safe Ranking)
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

# 4. Extracting ALL Rules Sorted by Support
global_rules = df_fuzzy.groupby(elite_5, observed=False)['Outcome'].agg([
    ('Rule_Outcome', lambda x: x.mode()[0] if not x.mode().empty else None),
    ('Support', 'count')
]).reset_index()

# 5. Top 10 Rules with Highest Overall Support
top_10_global = global_rules.sort_values('Support', ascending=False).head(10).reset_index(drop=True)

# 6. Output Table
print(f"{'Rank':<5} | {'Global Rule (Onset, ALSFRS, FVC, Weight, Speech)':<60} | {'Outcome':<8} | {'Support'}")
print("-" * 110)

for i, row in top_10_global.iterrows():
    combo = f"IF ({row.iloc[0]}, {row.iloc[1]}, {row.iloc[2]}, {row.iloc[3]}, {row.iloc[4]})"
    indicator = "🟢" if row['Rule_Outcome'] == 'SLOW' else ("🟡" if row['Rule_Outcome'] == 'MEDIUM' else "🔴")
    print(f"{i+1:<5} | {combo:<60} | {indicator} {row['Rule_Outcome']:<6} | {row['Support']} patients")

# Analysis of Coverage
total_top_support = top_10_global['Support'].sum()
print("\n" + "="*50)
print(f"Top 10 Rules cover {total_top_support} patients out of {len(data)} total.")
print(f"Coverage Ratio: {(total_top_support/len(data))*100:.2f}%")
print("="*50)