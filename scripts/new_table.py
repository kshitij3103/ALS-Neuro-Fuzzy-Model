import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SLOW_THRESHOLD = -0.5
FAST_THRESHOLD = -1.0

def get_outcome_label(v):
    if   v > SLOW_THRESHOLD: return "SLOW"
    elif v > FAST_THRESHOLD: return "MEDIUM"
    else:                    return "FAST"

def extract_rules_from_saved_model():
    path    = 'data/Preprocessed-data/trained_anfis_ensemble.pth'
    raw_csv = 'data/Preprocessed-data/MASTER_RAW_DATA.csv'

    checkpoint = torch.load(path, weights_only=False)
    raw_df     = pd.read_csv(raw_csv)

    sc_x          = checkpoint['scaler_x']
    sc_y          = checkpoint['scaler_y']
    features      = checkpoint['feature_names']
    member_models = checkpoint['member_models']
    n_clusters    = checkpoint['n_clusters']
    n_features    = len(features)

    all_centers     = np.stack([m['centers'].numpy()           for m in member_models])
    all_consequents = np.stack([m['consequent_params'].numpy() for m in member_models])
    mean_centers     = all_centers.mean(axis=0)
    mean_consequents = all_consequents.mean(axis=0)

    centers_raw  = sc_x.inverse_transform(mean_centers)
    out_scaled   = mean_consequents.reshape(n_clusters, n_features + 1)[:, -1]
    outcomes_raw = sc_y.inverse_transform(out_scaled.reshape(-1, 1)).flatten()

    # Per-feature thresholds
    thresholds = {}
    for feat in features:
        col = raw_df[feat].dropna()
        thresholds[feat] = (col.quantile(0.33), col.quantile(0.66))

    def label(val, feat):
        q33, q66 = thresholds[feat]
        if   val <= q33: return f"Low\n(<{q33:.1f})"
        elif val <= q66: return f"Medium\n({q33:.1f}-{q66:.1f})"
        else:            return f"High\n(>{q66:.1f})"

    # Patient support
    X_raw = raw_df[features].values.astype(float)
    meds  = np.nanmedian(X_raw, axis=0)
    for c in range(X_raw.shape[1]):
        X_raw[np.isnan(X_raw[:, c]), c] = meds[c]
    X_sc    = sc_x.transform(X_raw)
    dists   = np.linalg.norm(X_sc[:, np.newaxis] - mean_centers, axis=2)
    support = np.bincount(np.argmin(dists, axis=1), minlength=n_clusters)

    # Build sorted table
    rows = []
    for i in range(n_clusters):
        row = []
        for j, feat in enumerate(features):
            row.append(label(centers_raw[i][j], feat))
        row.append(get_outcome_label(outcomes_raw[i]))
        row.append(int(support[i]))
        rows.append(row)

    col_names = features + ["Outcome", "Support"]
    df = (pd.DataFrame(rows, columns=col_names)
            .sort_values("Support", ascending=False)
            .reset_index(drop=True))
    df.insert(0, "Rank", range(1, len(df) + 1))

    # ==========================================
    # DRAW TABLE USING MATPLOTLIB TABLE
    # ==========================================
    col_headers = ["Rank", "Onset\nDuration", "ALSFRS\nMedian",
                   "FVC\nSlope", "Weight\nSlope", "Speech\n(Q1)",
                   "Outcome", "Support"]

    display_cols = ["Rank"] + features + ["Outcome", "Support"]
    cell_data    = df[display_cols].values.tolist()

    outcome_colors = {"SLOW": "#90EE90", "MEDIUM": "#FFD580", "FAST": "#FF9999"}
    header_bg      = "#2C3E50"
    row_bg         = ["#FFFFFF", "#F0F3F4"]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    fig.suptitle("Table 4: Top 10 Data-Driven Fuzzy Rules by Patient Support (ALS Study)",
                 fontsize=14, fontweight='bold', y=0.98)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        loc='center',
        cellLoc='center'
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 3.2)   # scale(width_factor, height_factor) — height=3.2 gives plenty of row space

    n_rows = len(cell_data)
    n_cols = len(col_headers)

    # Style header row
    for ci in range(n_cols):
        cell = tbl[0, ci]
        cell.set_facecolor(header_bg)
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        cell.set_edgecolor('#AAAAAA')

    # Style data rows
    outcome_col_idx = col_headers.index("Outcome")
    for ri in range(1, n_rows + 1):
        bg = row_bg[(ri - 1) % 2]
        for ci in range(n_cols):
            cell = tbl[ri, ci]
            cell.set_edgecolor('#CCCCCC')
            if ci == outcome_col_idx:
                outcome_val = cell_data[ri - 1][outcome_col_idx]
                cell.set_facecolor(outcome_colors.get(outcome_val, bg))
                cell.set_text_props(fontweight='bold', fontsize=9)
            else:
                cell.set_facecolor(bg)
                cell.set_text_props(fontsize=9)

    # Column widths
    col_widths_map = {
        0: 0.04,   # Rank
        1: 0.13,   # Onset Duration
        2: 0.13,   # ALSFRS Median
        3: 0.11,   # FVC Slope
        4: 0.11,   # Weight Slope
        5: 0.11,   # Speech
        6: 0.09,   # Outcome
        7: 0.07,   # Support
    }
    for ci, w in col_widths_map.items():
        for ri in range(n_rows + 1):
            tbl[ri, ci].set_width(w)

    out_path = 'data/Preprocessed-data/Clinical_Rules_Table.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Table saved to: {out_path}")


if __name__ == "__main__":
    try:
        extract_rules_from_saved_model()
    except FileNotFoundError as e:
        print(f"File not found: {e}\nRun training script first.")
    except Exception as e:
        import traceback
        traceback.print_exc()