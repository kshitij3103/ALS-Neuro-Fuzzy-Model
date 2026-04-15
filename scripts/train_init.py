import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer


def pcc_scorer(y_true, y_pred):
    if len(np.unique(y_pred)) < 2: return 0
    return pearsonr(y_true, y_pred)[0]

pcc_cv_score = make_scorer(pcc_scorer)


data = pd.read_csv('data/Preprocessed-data/MASTER_RAW_DATA.csv')
X = data.drop(columns=['subject_id', 'target_slope'])
y = data['target_slope']

# 80-20 Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Selectors (Top 30 features )
k_feat = 30
selectors = {
    "Pearson (f_regression)": SelectKBest(score_func=f_regression, k=k_feat),
    "Mutual Information": SelectKBest(score_func=mutual_info_regression, k=k_feat),
    "RF Importance": SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), max_features=k_feat),
    "RFE (Recursive)": RFE(estimator=LinearRegression(), n_features_to_select=k_feat)
}

#  Define Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "Linear Regression": LinearRegression()
}

results = []

print(f"Starting Tournament: {len(selectors)} Selectors x {len(models)} Models...")


for s_name, selector in selectors.items():
    for m_name, model in models.items():
        print(f"Testing: {s_name} + {m_name}...")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler()),            
            ('selector', selector),
            ('model', model)
        ])
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42) 
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, 
                                scoring={'pcc': pcc_cv_score, 'rmse': 'neg_root_mean_squared_error'})
        
        results.append({
            "Selector": s_name,
            "Model": m_name,
            "Avg_PCC": np.mean(cv_res['test_pcc']),
            "Avg_RMSD": -np.mean(cv_res['test_rmse'])
        })


comparison_df = pd.DataFrame(results).sort_values(by="Avg_PCC", ascending=False)
print("\n--- FINAL TOURNAMENT RANKING ---")
print(comparison_df)

comparison_df.to_csv('data/Preprocessed-data/selector_model_tournament.csv', index=False)