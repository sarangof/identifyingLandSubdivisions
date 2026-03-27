"""
==========================================================================
Full Cascade Modeling Pipeline
==========================================================================

Runs all four stages of the classification cascade, plus:
- Gradient Boosted Tree comparison for Stage 3
- Precision / Recall / F1 at multiple thresholds
- Composite features (Option B: averaged correlated pairs)

Usage:
    from cascade_models import run_full_cascade
    results = run_full_cascade(blocks_labeled)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold
)
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, classification_report,
    roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance


# ======================================================================
# METRIC DEFINITIONS
# ======================================================================

ALL_13 = [
    'm1_std', 'm2_std', 'm3_std', 'm4_std', 'm5_std', 'm6_std',
    'm7_std', 'm8_std', 'm9_std', 'k_complexity_std',
    'm10_std', 'm11_std', 'm12_std'
]

REGULARITY_10 = [
    'm1_std', 'm2_std', 'm3_std', 'm4_std', 'm5_std', 'm6_std',
    'm7_std', 'm8_std', 'm9_std', 'k_complexity_std'
]

LABELS = {
    'm1_std': 'M1: bldg near road', 'm2_std': 'M2: avg bldg-road dist',
    'm3_std': 'M3: road density', 'm4_std': 'M4: 4-way intxn share',
    'm5_std': 'M5: intxn density', 'm6_std': 'M6: orientation KL',
    'm7_std': 'M7: block width', 'm8_std': 'M8: tortuosity',
    'm9_std': 'M9: angle deviation', 'k_complexity_std': 'K: complexity',
    'm10_std': 'M10: bldg density', 'm11_std': 'M11: built fraction',
    'm12_std': 'M12: avg bldg size',
    'road_access': 'Composite: road access (M1+M2)/2',
    'network_scale': 'Composite: network scale (M3+M7)/2',
}

# Composite features (Option B)
COMPOSITE_8 = [
    'road_access', 'network_scale', 'm4_std', 'm5_std', 'm6_std',
    'm8_std', 'm9_std', 'k_complexity_std'
]


def add_composite_features(df):
    """Add averaged correlated-pair composites."""
    df = df.copy()
    df['road_access'] = (df['m1_std'] + df['m2_std']) / 2
    df['network_scale'] = (df['m3_std'] + df['m7_std']) / 2
    return df


# ======================================================================
# CORE: calibrated model evaluation for one stage
# ======================================================================

def evaluate_stage(df, feature_cols, target_col, positive_class,
                   stage_name, n_splits=10, n_repeats=5):
    """
    Run calibrated logistic regression for one cascade stage.
    Returns coefficients, AUC, calibrated probabilities, precision/recall/F1.
    """
    data = df[feature_cols + [target_col]].dropna().copy()
    y = (data[target_col] == positive_class).astype(int).values
    X = data[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Repeated k-fold for coefficient stability ──
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    all_coefs = []
    all_aucs = []

    for train_idx, test_idx in rskf.split(X_scaled, y):
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=5000, class_weight='balanced', C=1.0
        )
        model.fit(X_scaled[train_idx], y[train_idx])
        prob = model.predict_proba(X_scaled[test_idx])[:, 1]
        all_coefs.append(model.coef_[0])
        all_aucs.append(roc_auc_score(y[test_idx], prob))

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

    # ── Calibrated CV for honest probabilities ──
    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = np.full(len(y), np.nan)

    for train_idx, test_idx in cv_eval.split(X_scaled, y):
        fold_model = CalibratedClassifierCV(
            estimator=LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5,
                max_iter=5000, class_weight='balanced', C=1.0
            ),
            method='isotonic', cv=3
        )
        fold_model.fit(X_scaled[train_idx], y[train_idx])
        y_prob_cv[test_idx] = fold_model.predict_proba(X_scaled[test_idx])[:, 1]

    auc_cal = roc_auc_score(y, y_prob_cv)
    brier = brier_score_loss(y, y_prob_cv)
    prob_true, prob_pred = calibration_curve(y, y_prob_cv, n_bins=10, strategy='quantile')
    ece = np.mean(np.abs(prob_true - prob_pred))

    # ── Precision / Recall / F1 at multiple thresholds ──
    thresholds_to_report = [0.3, 0.4, 0.5, 0.6, 0.7]
    pr_table = []
    for t in thresholds_to_report:
        y_pred_t = (y_prob_cv >= t).astype(int)
        pr_table.append({
            'Threshold': t,
            'Precision_pos': precision_score(y, y_pred_t, zero_division=0),
            'Recall_pos': recall_score(y, y_pred_t, zero_division=0),
            'F1_pos': f1_score(y, y_pred_t, zero_division=0),
            'Precision_neg': precision_score(y, y_pred_t, pos_label=0, zero_division=0),
            'Recall_neg': recall_score(y, y_pred_t, pos_label=0, zero_division=0),
            'F1_neg': f1_score(y, y_pred_t, pos_label=0, zero_division=0),
        })
    pr_df = pd.DataFrame(pr_table)

    # ── Print ──
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"  N = {len(y):,}  ({n_pos:,} positive, {n_neg:,} negative, ratio {n_pos/n_neg:.1f}:1)")
    print(f"  ROC-AUC (repeated k-fold): {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"  ROC-AUC (calibrated):      {auc_cal:.4f}")
    print(f"  Brier score:               {brier:.4f}")
    print(f"  ECE:                       {ece:.4f}")
    print(f"\n  Coefficients (regularity model):")
    for _, r in coef_summary.iterrows():
        print(f"    {r['Label']:>35s}: {r['Mean_Coef']:+.3f} [{r['CI_lower']:+.3f}, {r['CI_upper']:+.3f}]")
    print(f"\n  Precision / Recall / F1 at different thresholds:")
    print(f"  (positive class = {positive_class})")
    print(pr_df.to_string(index=False))

    # ── Calibration plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    ax1.plot(prob_pred, prob_true, 'o-', color='#534AB7', linewidth=2, label='Model')
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Observed frequency')
    ax1.set_title(f'Reliability diagram — {stage_name}')
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(y_prob_cv[y == 1], bins=30, alpha=0.5, color='#1D9E75', label=positive_class, density=True)
    ax2.hist(y_prob_cv[y == 0], bins=30, alpha=0.5, color='#D85A30', label=f'not {positive_class}', density=True)
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Score distributions — {stage_name}')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'cascade_{stage_name.replace(" ", "_").replace("/","_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Train a final calibrated model on ALL data for scoring ──
    final_model = CalibratedClassifierCV(
        estimator=LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=5000, class_weight='balanced', C=1.0
        ),
        method='isotonic', cv=5
    )
    final_model.fit(X_scaled, y)

    return {
        'stage_name': stage_name,
        'feature_cols': feature_cols,
        'coef_summary': coef_summary,
        'coefs_array': coefs_array,
        'aucs_repeated': all_aucs,
        'auc_cal': auc_cal,
        'brier': brier,
        'ece': ece,
        'y_true': y,
        'y_prob_cv': y_prob_cv,
        'calibration': (prob_true, prob_pred),
        'pr_table': pr_df,
        'scaler': scaler,
        'model': final_model,
    }


# ======================================================================
# GRADIENT BOOSTED TREE COMPARISON (Stage 3 only)
# ======================================================================

def run_gbt_comparison(df, feature_cols, target_col, positive_class, lr_result):
    """
    Run gradient boosted trees on the same data as Stage 3
    to compare against logistic regression.
    """
    data = df[feature_cols + [target_col]].dropna().copy()
    y = (data[target_col] == positive_class).astype(int).values
    X = data[feature_cols].values

    # No scaling needed for tree models
    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── GBT with calibration ──
    y_prob_gbt = np.full(len(y), np.nan)
    for train_idx, test_idx in cv_eval.split(X, y):
        gbt = CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.1,
                class_weight='balanced', random_state=42
            ),
            method='isotonic', cv=3
        )
        gbt.fit(X[train_idx], y[train_idx])
        y_prob_gbt[test_idx] = gbt.predict_proba(X[test_idx])[:, 1]

    auc_gbt = roc_auc_score(y, y_prob_gbt)
    brier_gbt = brier_score_loss(y, y_prob_gbt)

    # ── Feature importance (from uncalibrated GBT on full data) ──
    gbt_full = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1,
        class_weight='balanced', random_state=42
    )
    gbt_full.fit(X, y)
    

    perm = permutation_importance(gbt_full, X, y, n_repeats=10, random_state=42, scoring='roc_auc')
    importances = pd.DataFrame({
        'Metric': feature_cols,
        'Label': [LABELS.get(m, m) for m in feature_cols],
        'Importance': perm.importances_mean,
    }).sort_values('Importance', ascending=False)

    # ── Comparison plot ──
    auc_lr = lr_result['auc_cal']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC comparison
    ax = axes[0]
    fpr_lr, tpr_lr, _ = roc_curve(lr_result['y_true'], lr_result['y_prob_cv'])
    fpr_gbt, tpr_gbt, _ = roc_curve(y, y_prob_gbt)
    ax.plot(fpr_lr, tpr_lr, color='#534AB7', linewidth=2, label=f'Logistic Reg (AUC={auc_lr:.3f})')
    ax.plot(fpr_gbt, tpr_gbt, color='#1D9E75', linewidth=2, label=f'Grad Boosted (AUC={auc_gbt:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC: Logistic Regression vs Gradient Boosted Trees')
    ax.legend()

    # Feature importance (GBT)
    ax = axes[1]
    imp = importances.sort_values('Importance', ascending=True)
    ax.barh(range(len(imp)), imp['Importance'], color='#1D9E75', alpha=0.7)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['Label'])
    ax.set_xlabel('Feature importance')
    ax.set_title('GBT feature importance')

    # Summary text
    ax = axes[2]
    ax.axis('off')
    summary_text = (
        f"Logistic Regression (calibrated)\n"
        f"  AUC:   {auc_lr:.4f}\n"
        f"  Brier: {lr_result['brier']:.4f}\n"
        f"  ECE:   {lr_result['ece']:.4f}\n\n"
        f"Gradient Boosted Trees (calibrated)\n"
        f"  AUC:   {auc_gbt:.4f}\n"
        f"  Brier: {brier_gbt:.4f}\n\n"
        f"AUC difference: {auc_gbt - auc_lr:+.4f}\n\n"
    )
    if abs(auc_gbt - auc_lr) < 0.02:
        summary_text += "→ Small gap: linear model captures\n  most of the available signal.\n  Logistic regression is sufficient."
    else:
        summary_text += "→ Notable gap: nonlinear interactions\n  may be present. Investigate further."
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    ax.set_title('Model comparison summary')

    plt.tight_layout()
    plt.savefig('gbt_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON: Logistic Regression vs Gradient Boosted Trees")
    print(f"{'='*60}")
    print(f"  Logistic Regression AUC: {auc_lr:.4f}")
    print(f"  Gradient Boosted AUC:    {auc_gbt:.4f}")
    print(f"  Difference:              {auc_gbt - auc_lr:+.4f}")
    print(f"\n  GBT Feature Importance:")
    for _, r in importances.iterrows():
        print(f"    {r['Label']:>35s}: {r['Importance']:.3f}")

    return {
        'auc_gbt': auc_gbt,
        'brier_gbt': brier_gbt,
        'y_prob_gbt': y_prob_gbt,
        'importances': importances,
    }


# ======================================================================
# OPEN SPACE ANALYSIS
# ======================================================================

def analyze_open_space(blocks_labeled):
    """
    Document why open space classification doesn't need a model.
    """
    print(f"\n{'='*60}")
    print(f"  OPEN SPACE ANALYSIS")
    print(f"{'='*60}")

    built = blocks_labeled[blocks_labeled['built_vs_open'] == 'built_up']
    open_sp = blocks_labeled[blocks_labeled['built_vs_open'] == 'open_space']

    print(f"\n  Built-up blocks: {len(built):,}")
    print(f"  Open space blocks: {len(open_sp):,}")
    print(f"  Ratio: {len(built)/len(open_sp):.0f}:1")

    print(f"\n  Open space blocks with n_buildings > 0: {(open_sp['n_buildings'] > 0).sum()}/{len(open_sp)}")
    print(f"  (All open space blocks have buildings — n_buildings filter was applied upstream)")

    # Compare distributions
    for col in ['n_buildings', 'm10_raw', 'm11_raw', 'block_area']:
        b_med = built[col].median()
        o_med = open_sp[col].median()
        print(f"\n  {col}:")
        print(f"    Built-up median: {b_med:.1f}")
        print(f"    Open space median: {o_med:.1f}")

    # The argument: building density is dramatically different
    print(f"\n  Building density (M10 raw) distribution:")
    print(f"    Built-up:    25th={built['m10_raw'].quantile(0.25):.0f}, "
          f"50th={built['m10_raw'].quantile(0.5):.0f}, "
          f"75th={built['m10_raw'].quantile(0.75):.0f}")
    print(f"    Open space:  25th={open_sp['m10_raw'].quantile(0.25):.0f}, "
          f"50th={open_sp['m10_raw'].quantile(0.5):.0f}, "
          f"75th={open_sp['m10_raw'].quantile(0.75):.0f}")

    print(f"\n  Recommendation:")
    print(f"  With only {len(open_sp)} open space blocks (vs {len(built):,} built-up),")
    print(f"  a formal model is unreliable. Open space detection in production")
    print(f"  should use n_buildings (from the block metrics pipeline) combined")
    print(f"  with building density thresholds. For the regularity index,")
    print(f"  blocks with n_buildings = 0 are excluded upstream, and the")
    print(f"  remaining blocks proceed to the residential classification stage.")

    # Run the model anyway for documentation
    print(f"\n  Running logistic model for documentation (not recommended for production):")
    stage1 = evaluate_stage(
        df=blocks_labeled,
        feature_cols=['m3_std', 'm4_std', 'm5_std', 'm7_std', 'm10_std', 'm11_std', 'k_complexity_std'],
        target_col='built_vs_open',
        positive_class='built_up',
        stage_name='Stage 1: Built vs Open',
        n_repeats=3  # fewer repeats since this is just for documentation
    )

    return stage1


# ======================================================================
# MAIN: RUN FULL CASCADE
# ======================================================================

def run_full_cascade(blocks_labeled):
    """
    Run all four cascade stages + GBT comparison + composite features.
    """
    # Add composite features
    blocks_labeled = add_composite_features(blocks_labeled)

    # ── Stage 1: Open space ──
    stage1 = analyze_open_space(blocks_labeled)

    # ── Stage 2: Residential vs non-residential ──
    df_built = blocks_labeled[blocks_labeled['built_vs_open'] == 'built_up'].copy()
    print(f"\n  Built-up blocks for Stage 2: {len(df_built):,}")

    stage2 = evaluate_stage(
        df=df_built,
        feature_cols=ALL_13,  # saturation metrics appropriate here
        target_col='residential',
        positive_class='residential',
        stage_name='Stage 2: Residential vs Non-residential'
    )

    # ── Stage 3: Subdivision vs irregular (CORE MODEL) ──
    df_res = df_built[df_built['residential'] == 'residential'].copy()
    print(f"\n  Residential blocks for Stage 3: {len(df_res):,}")

    # 3a: With 10 regularity metrics
    stage3_reg = evaluate_stage(
        df=df_res,
        feature_cols=REGULARITY_10,
        target_col='subdivisions_settlements',
        positive_class='subdivision',
        stage_name='Stage 3a: Subdivision vs Irregular (10 regularity metrics)'
    )

    # 3b: With composite features (Option B — stable coefficients)
    stage3_composite = evaluate_stage(
        df=df_res,
        feature_cols=COMPOSITE_8,
        target_col='subdivisions_settlements',
        positive_class='subdivision',
        stage_name='Stage 3b: Subdivision vs Irregular (8 composite metrics)'
    )

    # ── Gradient Boosted Tree comparison (Stage 3) ──
    gbt_results = run_gbt_comparison(
        df=df_res,
        feature_cols=REGULARITY_10,
        target_col='subdivisions_settlements',
        positive_class='subdivision',
        lr_result=stage3_reg
    )

    # ── Stage 4: Formal vs informal ──
    df_sub = df_res[df_res['subdivisions_settlements'] == 'subdivision'].copy()
    print(f"\n  Subdivision blocks for Stage 4: {len(df_sub):,}")

    stage4 = evaluate_stage(
        df=df_sub,
        feature_cols=REGULARITY_10,
        target_col='formal_informal_subdivisions',
        positive_class='formal_subdivisions',
        stage_name='Stage 4: Formal vs Informal'
    )

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  CASCADE SUMMARY")
    print(f"{'='*70}")
    summary_data = []
    for name, res in [
        ('Stage 1: Built vs Open', stage1),
        ('Stage 2: Residential', stage2),
        ('Stage 3a: Regularity (10)', stage3_reg),
        ('Stage 3b: Composites (8)', stage3_composite),
        ('Stage 4: Formal/Informal', stage4),
    ]:
        summary_data.append({
            'Stage': name,
            'N features': len(res['feature_cols']),
            'AUC': f"{res['auc_cal']:.4f}",
            'Brier': f"{res['brier']:.4f}",
            'ECE': f"{res['ece']:.4f}",
        })
    summary_data.append({
        'Stage': 'GBT comparison (Stage 3)',
        'N features': len(REGULARITY_10),
        'AUC': f"{gbt_results['auc_gbt']:.4f}",
        'Brier': f"{gbt_results['brier_gbt']:.4f}",
        'ECE': '—',
    })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    return {
        'stage1': stage1,
        'stage2': stage2,
        'stage3_reg': stage3_reg,
        'stage3_composite': stage3_composite,
        'stage4': stage4,
        'gbt': gbt_results,
        'summary': summary_df,
    }

if __name__ == "__main__":
    print("""
    Usage:
        from cascade_models import run_full_cascade
        results = run_full_cascade(blocks_labeled)
    """)