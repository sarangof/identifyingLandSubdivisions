"""
==========================================================================
Model Comparison: Full metrics vs Regularity-only (excluding saturation)
==========================================================================

Saturation metrics M10, M11, M12 measure *how much* a block is built up,
not *how regularly* it was planned. This script compares:

  Model A: All 13 metrics (current)
  Model B: 10 regularity metrics (excluding M10, M11, M12)

For Stage 3 (subdivision vs irregular) — the core regularity model.

Also re-examines K-complexity which showed a clear univariate signal
(mean 2.47 irregular vs 1.89 subdivision) but was washed out by
multicollinearity with M10/M11 in the full model.

Run AFTER cells 10+59 in validation.ipynb (blocks_labeled exists with _std cols).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             classification_report, roc_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


# ======================================================================
# CONFIG
# ======================================================================

ALL_METRICS = [
    'm1_std', 'm2_std', 'm3_std', 'm4_std', 'm5_std', 'm6_std',
    'm7_std', 'm8_std', 'm9_std', 'k_complexity_std',
    'm10_std', 'm11_std', 'm12_std'
]

REGULARITY_ONLY = [
    'm1_std', 'm2_std', 'm3_std', 'm4_std', 'm5_std', 'm6_std',
    'm7_std', 'm8_std', 'm9_std', 'k_complexity_std'
]

LABELS = {
    'm1_std':  'M1: bldg near road',
    'm2_std':  'M2: avg bldg-road dist',
    'm3_std':  'M3: road density',
    'm4_std':  'M4: 4-way intxn share',
    'm5_std':  'M5: intxn density',
    'm6_std':  'M6: orientation KL',
    'm7_std':  'M7: block width',
    'm8_std':  'M8: tortuosity',
    'm9_std':  'M9: angle deviation',
    'k_complexity_std': 'K: complexity',
    'm10_std': 'M10: bldg density',
    'm11_std': 'M11: built fraction',
    'm12_std': 'M12: avg bldg size',
}


# ======================================================================
# CORE: run one model variant with full diagnostics
# ======================================================================

def run_model_variant(df, feature_cols, target_col, positive_class, label=""):
    """
    Runs repeated k-fold for coefficient stability + calibrated evaluation.
    Returns dict with all diagnostics.
    """
    data = df[feature_cols + [target_col]].dropna().copy()
    y = (data[target_col] == positive_class).astype(int)
    X = data[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_arr = X_scaled
    y_arr = y.values

    # ---- Repeated k-fold for coefficient stability ----
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    all_coefs = []
    all_aucs = []

    for train_idx, test_idx in rskf.split(X_arr, y_arr):
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=5000, class_weight='balanced', C=1.0
        )
        model.fit(X_arr[train_idx], y_arr[train_idx])
        y_prob = model.predict_proba(X_arr[test_idx])[:, 1]
        all_coefs.append(model.coef_[0])
        all_aucs.append(roc_auc_score(y_arr[test_idx], y_prob))

    coefs_array = np.array(all_coefs)

    coef_summary = pd.DataFrame({
        'Metric': feature_cols,
        'Label': [LABELS.get(m, m) for m in feature_cols],
        'Mean_Coef': coefs_array.mean(axis=0),
        'Std_Coef': coefs_array.std(axis=0),
        'CI_lower': np.percentile(coefs_array, 2.5, axis=0),
        'CI_upper': np.percentile(coefs_array, 97.5, axis=0),
    })
    coef_summary['Abs_Mean'] = coef_summary['Mean_Coef'].abs()
    coef_summary = coef_summary.sort_values('Abs_Mean', ascending=False)

    # ---- Calibrated CV evaluation ----
    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = np.full(len(y_arr), np.nan)

    for train_idx, test_idx in cv_eval.split(X_arr, y_arr):
        fold_model = CalibratedClassifierCV(
            estimator=LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5,
                max_iter=5000, class_weight='balanced', C=1.0
            ),
            method='isotonic', cv=3
        )
        fold_model.fit(X_arr[train_idx], y_arr[train_idx])
        y_prob_cv[test_idx] = fold_model.predict_proba(X_arr[test_idx])[:, 1]

    auc_calibrated = roc_auc_score(y_arr, y_prob_cv)
    brier = brier_score_loss(y_arr, y_prob_cv)
    prob_true, prob_pred = calibration_curve(y_arr, y_prob_cv, n_bins=10, strategy='quantile')
    ece = np.mean(np.abs(prob_true - prob_pred))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Features:           {len(feature_cols)}")
    print(f"  ROC-AUC (repeated): {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"  ROC-AUC (calib):    {auc_calibrated:.4f}")
    print(f"  Brier score:        {brier:.4f}")
    print(f"  ECE:                {ece:.4f}")
    print(f"\n  Coefficients:")
    for _, row in coef_summary.iterrows():
        print(f"    {row['Label']:>25s}: {row['Mean_Coef']:+.3f} [{row['CI_lower']:+.3f}, {row['CI_upper']:+.3f}]")

    return {
        'label': label,
        'feature_cols': feature_cols,
        'coef_summary': coef_summary,
        'coefs_array': coefs_array,
        'aucs_repeated': all_aucs,
        'auc_calibrated': auc_calibrated,
        'brier': brier,
        'ece': ece,
        'y_true': y_arr,
        'y_prob_cv': y_prob_cv,
        'calibration': (prob_true, prob_pred),
    }


# ======================================================================
# COMPARISON PLOTS
# ======================================================================

def plot_comparison(result_a, result_b):
    """
    Side-by-side comparison of two model variants.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- 1. ROC curves ----
    ax = axes[0, 0]
    for res, color, ls in [(result_a, '#534AB7', '-'), (result_b, '#D85A30', '--')]:
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob_cv'])
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                label=f"{res['label']} (AUC={res['auc_calibrated']:.3f})")
    ax.plot([0,1], [0,1], 'k:', linewidth=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curves')
    ax.legend(fontsize=9)

    # ---- 2. Calibration curves ----
    ax = axes[0, 1]
    for res, color, ls in [(result_a, '#534AB7', '-'), (result_b, '#D85A30', '--')]:
        pt, pp = res['calibration']
        ax.plot(pp, pt, 'o-', color=color, linestyle=ls, linewidth=2,
                label=f"{res['label']} (ECE={res['ece']:.3f})")
    ax.plot([0,1], [0,1], 'k:', linewidth=0.5)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Calibration comparison')
    ax.legend(fontsize=9)

    # ---- 3. Coefficient comparison (regularity-only model) ----
    ax = axes[1, 0]
    cs = result_b['coef_summary'].sort_values('Abs_Mean', ascending=True)
    colors = ['#534AB7' if v > 0 else '#D85A30' for v in cs['Mean_Coef']]
    ax.barh(range(len(cs)), cs['Mean_Coef'], color=colors, alpha=0.7)
    ax.errorbar(cs['Mean_Coef'], range(len(cs)),
                xerr=[cs['Mean_Coef']-cs['CI_lower'], cs['CI_upper']-cs['Mean_Coef']],
                fmt='none', color='black', capsize=3, linewidth=0.8)
    ax.set_yticks(range(len(cs)))
    ax.set_yticklabels([LABELS.get(m, m) for m in cs['Metric']])
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('Mean coefficient (standardized)')
    ax.set_title(f'Coefficients: {result_b["label"]}')

    # ---- 4. AUC distributions ----
    ax = axes[1, 1]
    bp = ax.boxplot(
        [result_a['aucs_repeated'], result_b['aucs_repeated']],
        labels=[result_a['label'], result_b['label']],
        patch_artist=True, widths=0.5
    )
    bp['boxes'][0].set_facecolor('#534AB7')
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor('#D85A30')
    bp['boxes'][1].set_alpha(0.4)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('AUC distribution across 50 folds')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---- Summary table ----
    summary = pd.DataFrame({
        'Model': [result_a['label'], result_b['label']],
        'N_features': [len(result_a['feature_cols']), len(result_b['feature_cols'])],
        'AUC_mean': [np.mean(result_a['aucs_repeated']), np.mean(result_b['aucs_repeated'])],
        'AUC_std': [np.std(result_a['aucs_repeated']), np.std(result_b['aucs_repeated'])],
        'AUC_calibrated': [result_a['auc_calibrated'], result_b['auc_calibrated']],
        'Brier': [result_a['brier'], result_b['brier']],
        'ECE': [result_a['ece'], result_b['ece']],
    })
    print("\n" + "="*70)
    print("  MODEL COMPARISON SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))

    # AUC difference significance
    auc_diff = np.array(result_a['aucs_repeated']) - np.array(result_b['aucs_repeated'])
    mean_diff = auc_diff.mean()
    se_diff = auc_diff.std() / np.sqrt(len(auc_diff))
    t_stat = mean_diff / se_diff if se_diff > 0 else 0
    print(f"\n  Paired AUC difference (A - B): {mean_diff:+.4f} ± {auc_diff.std():.4f}")
    print(f"  t-statistic: {t_stat:.2f} (>2 = significant at p<0.05)")

    return summary


# ======================================================================
# REGULARIZATION PATH for regularity-only model
# ======================================================================

def plot_reg_path_regularity(df, feature_cols, target_col, positive_class, n_alphas=60):
    """Regularization path for the regularity-only model."""
    data = df[feature_cols + [target_col]].dropna()
    y = (data[target_col] == positive_class).astype(int)
    X = data[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    C_values = np.logspace(-3, 2, n_alphas)
    coefs = {col: [] for col in feature_cols}

    for C in C_values:
        model = LogisticRegression(
            penalty='l1', solver='saga', C=C,
            max_iter=5000, class_weight='balanced'
        )
        model.fit(X_scaled, y)
        for j, col in enumerate(feature_cols):
            coefs[col].append(model.coef_[0][j])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

    for i, col in enumerate(feature_cols):
        label = LABELS.get(col, col).split(':')[0]
        ax.plot(C_values, coefs[col], label=label, linewidth=1.5, color=colors[i])

    ax.set_xscale('log')
    ax.set_xlabel('C (inverse regularization) → weaker penalty →')
    ax.set_ylabel('Coefficient value')
    ax.set_title('Regularization path (L1) — Regularity metrics only (no M10, M11, M12)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig('regularization_path_regularity_only.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Entry order
    entry_order = {}
    for col in feature_cols:
        for j, C in enumerate(C_values):
            if abs(coefs[col][j]) > 1e-6:
                entry_order[col] = C
                break
        else:
            entry_order[col] = np.inf

    entry_df = pd.DataFrame({
        'Metric': list(entry_order.keys()),
        'Label': [LABELS.get(m, m) for m in entry_order.keys()],
        'Entry_C': list(entry_order.values())
    }).sort_values('Entry_C')
    entry_df['Rank'] = range(1, len(entry_df) + 1)
    print("\nFeature entry order (regularity-only):")
    print(entry_df.to_string(index=False))

    return coefs, entry_df


# ======================================================================
# MAIN
# ======================================================================

def run_comparison(blocks_labeled):
    """
    Main entry point. Call with blocks_labeled from your notebook.
    """
    # Filter to residential blocks
    df_built = blocks_labeled[blocks_labeled['built_vs_open'] == 'built_up'].copy()
    df_res = df_built[df_built['residential'] == 'residential'].copy()

    print(f"Residential blocks: {len(df_res):,}")
    print(f"  Subdivisions:         {(df_res['subdivisions_settlements']=='subdivision').sum():,}")
    print(f"  Irregular settlements: {(df_res['subdivisions_settlements']=='irregular_settlement').sum():,}")

    # ---- Model A: All 13 metrics ----
    result_a = run_model_variant(
        df=df_res,
        feature_cols=ALL_METRICS,
        target_col='subdivisions_settlements',
        positive_class='subdivision',
        label='All 13 metrics'
    )

    # ---- Model B: Regularity only (no M10, M11, M12) ----
    result_b = run_model_variant(
        df=df_res,
        feature_cols=REGULARITY_ONLY,
        target_col='subdivisions_settlements',
        positive_class='subdivision',
        label='Regularity only (10 metrics)'
    )

    # ---- Comparison plots ----
    summary = plot_comparison(result_a, result_b)

    # ---- Regularization path for regularity-only ----
    plot_reg_path_regularity(
        df=df_res,
        feature_cols=REGULARITY_ONLY,
        target_col='subdivisions_settlements',
        positive_class='subdivision'
    )

    return result_a, result_b, summary


# ======================================================================
# USAGE
# ======================================================================

if __name__ == "__main__":
    print("""
    Usage in your notebook (after blocks_labeled is defined):

        from model_comparison import run_comparison
        result_a, result_b, summary = run_comparison(blocks_labeled)
    """)
