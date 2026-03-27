"""
==========================================================================
Urban Regularity Index — Dimensionality Analysis & Calibrated Cascade
==========================================================================

Pipeline for:
  1. PCA — how many independent signals exist in the 13 metrics?
  2. Factor Analysis — what are the latent constructs?
  3. Regularization paths — which metrics are robustly important?
  4. Repeated stratified k-fold — stable coefficients + confidence intervals
  5. Calibrated probability cascade — interpretable regularity index

Run this AFTER Cell 10 in validation.ipynb (where all_cities and
blocks_labeled are already loaded with _std columns computed).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold, cross_val_predict, cross_validate
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline

# factor_analyzer needs: pip install factor-analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


# ======================================================================
# CONFIG — adjust these to match your notebook state
# ======================================================================

metrics_std = [
    'm1_std', 'm2_std', 'm3_std', 'm4_std', 'm5_std', 'm6_std',
    'm7_std', 'm8_std', 'm9_std', 'k_complexity_std',
    'm10_std', 'm11_std', 'm12_std'
]

METRIC_LABELS = {
    'm1_std':  'M1: buildings within 20m of road',
    'm2_std':  'M2: avg building-road distance',
    'm3_std':  'M3: road density',
    'm4_std':  'M4: 4-way intersection share',
    'm5_std':  'M5: intersection density',
    'm6_std':  'M6: building orientation KL',
    'm7_std':  'M7: block width',
    'm8_std':  'M8: road tortuosity',
    'm9_std':  'M9: intersection angle deviation',
    'k_complexity_std': 'K: parcel-layer complexity',
    'm10_std': 'M10: building density',
    'm11_std': 'M11: built-up fraction',
    'm12_std': 'M12: avg building size',
}


# ======================================================================
# PART 1: PCA — How many independent signals?
# ======================================================================

def run_pca_analysis(df, feature_cols=metrics_std, title_suffix=""):
    """
    Full PCA diagnostic:
    - Scree plot with parallel analysis
    - Cumulative variance explained
    - Component loadings heatmap
    """
    X = df[feature_cols].dropna()
    print(f"PCA on {len(X):,} observations, {len(feature_cols)} features")

    # Standardize (PCA is scale-sensitive)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_
    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)

    # --- Parallel Analysis (random data comparison) ---
    n_iter = 100
    random_eigenvalues = np.zeros((n_iter, len(feature_cols)))
    for i in range(n_iter):
        random_data = np.random.normal(size=X_scaled.shape)
        random_pca = PCA()
        random_pca.fit(random_data)
        random_eigenvalues[i] = random_pca.explained_variance_

    random_95th = np.percentile(random_eigenvalues, 95, axis=0)

    # Number of components to retain (eigenvalue > random 95th percentile)
    n_retain = np.sum(eigenvalues > random_95th)
    print(f"\nParallel analysis suggests retaining {n_retain} components")
    print(f"(Kaiser criterion — eigenvalue > 1 — suggests {np.sum(eigenvalues > 1)})")

    # --- Figure 1: Scree plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    components = range(1, len(eigenvalues) + 1)
    ax1.plot(components, eigenvalues, 'o-', color='#534AB7', linewidth=2, label='Observed')
    ax1.plot(components, random_95th, 's--', color='#D85A30', linewidth=1.5, label='95th pctl random')
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='Kaiser criterion')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title(f'Scree plot with parallel analysis{title_suffix}')
    ax1.legend()
    ax1.set_xticks(list(components))

    ax2 = axes[1]
    ax2.bar(components, var_explained * 100, color='#534AB7', alpha=0.6, label='Individual')
    ax2.plot(components, cum_var * 100, 'o-', color='#D85A30', linewidth=2, label='Cumulative')
    ax2.axhline(y=80, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Variance explained (%)')
    ax2.set_title(f'Variance explained{title_suffix}')
    ax2.legend()
    ax2.set_xticks(list(components))

    plt.tight_layout()
    plt.savefig(f'pca_scree{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Figure 2: Loadings heatmap for retained components ---
    loadings = pd.DataFrame(
        pca.components_[:n_retain].T,
        index=feature_cols,
        columns=[f'PC{i+1}' for i in range(n_retain)]
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
        linewidths=0.5, ax=ax, vmin=-1, vmax=1
    )
    ax.set_title(f'PCA loadings (top {n_retain} components){title_suffix}')
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in feature_cols], rotation=0)
    plt.tight_layout()
    plt.savefig(f'pca_loadings{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Summary table ---
    summary = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(eigenvalues))],
        'Eigenvalue': eigenvalues,
        'Var_Explained_%': var_explained * 100,
        'Cumulative_%': cum_var * 100,
        'Random_95th': random_95th,
        'Retain': eigenvalues > random_95th
    })
    print("\n", summary.to_string(index=False))

    return pca, scaler, loadings, n_retain


# ======================================================================
# PART 2: Factor Analysis — What are the latent constructs?
# ======================================================================

def run_factor_analysis(df, n_factors, feature_cols=metrics_std, rotation='promax',
                        title_suffix=""):
    """
    Exploratory Factor Analysis with oblique rotation.
    - Bartlett's test of sphericity
    - KMO measure of sampling adequacy
    - Factor loadings heatmap
    - Factor correlation matrix (oblique rotation)
    """
    X = df[feature_cols].dropna()
    print(f"Factor Analysis on {len(X):,} observations, {n_factors} factors")

    # --- Adequacy tests ---
    chi_square, p_value = calculate_bartlett_sphericity(X)
    print(f"\nBartlett's test: chi² = {chi_square:.1f}, p = {p_value:.2e}")
    print("  (p < 0.05 means correlations are significantly different from identity → FA is appropriate)")

    kmo_all, kmo_model = calculate_kmo(X)
    print(f"\nKMO measure: {kmo_model:.3f}")
    if kmo_model >= 0.8:
        print("  Meritorious — excellent for FA")
    elif kmo_model >= 0.7:
        print("  Middling — acceptable for FA")
    elif kmo_model >= 0.6:
        print("  Mediocre — proceed with caution")
    else:
        print("  Poor — FA may not be appropriate")

    print(f"\nPer-variable KMO:")
    for col, kmo_val in zip(feature_cols, kmo_all):
        flag = " ⚠️ LOW" if kmo_val < 0.5 else ""
        print(f"  {METRIC_LABELS.get(col, col):>40s}: {kmo_val:.3f}{flag}")

    # --- Fit Factor Analysis ---
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='ml')
    fa.fit(X)

    # Loadings
    loadings = pd.DataFrame(
        fa.loadings_,
        index=feature_cols,
        columns=[f'Factor {i+1}' for i in range(n_factors)]
    )

    # --- Communalities (how much variance each variable shares with the factors) ---
    communalities = pd.DataFrame({
        'Variable': feature_cols,
        'Label': [METRIC_LABELS.get(m, m) for m in feature_cols],
        'Communality': fa.get_communalities(),
        'Uniqueness': fa.get_uniquenesses()
    })
    print("\nCommunalities (higher = better captured by the factors):")
    print(communalities.to_string(index=False))

    # --- Figure: Loadings heatmap ---
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
        linewidths=0.5, ax=ax, vmin=-1, vmax=1
    )
    ax.set_title(f'Factor loadings ({rotation} rotation, {n_factors} factors){title_suffix}')
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in feature_cols], rotation=0)
    plt.tight_layout()
    plt.savefig(f'fa_loadings{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Factor correlations (only meaningful for oblique rotation) ---
    if rotation in ('promax', 'oblimin', 'quartimin'):
        factor_corr = pd.DataFrame(
            fa.phi_,
            index=[f'Factor {i+1}' for i in range(n_factors)],
            columns=[f'Factor {i+1}' for i in range(n_factors)]
        )
        print(f"\nFactor correlations ({rotation}):")
        print(factor_corr.round(3).to_string())

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(factor_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
        ax.set_title(f'Factor correlation matrix')
        plt.tight_layout()
        plt.savefig(f'fa_correlations{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()

    # --- Dominant factor assignment for each metric ---
    dominant = loadings.abs().idxmax(axis=1)
    dominant_loading = loadings.abs().max(axis=1)
    assignment = pd.DataFrame({
        'Metric': feature_cols,
        'Label': [METRIC_LABELS.get(m, m) for m in feature_cols],
        'Dominant_Factor': dominant.values,
        'Loading': dominant_loading.values,
        'Cross_loading': ['YES' if (loadings.loc[m].abs() > 0.3).sum() > 1 else 'no'
                          for m in feature_cols]
    }).sort_values('Dominant_Factor')

    print("\nFactor assignments:")
    print(assignment.to_string(index=False))

    return fa, loadings, communalities


# ======================================================================
# PART 3: Regularization Path — Which metrics are robustly important?
# ======================================================================

def plot_regularization_path(df, feature_cols, target_col, positive_class,
                              n_alphas=60, title=""):
    """
    Plot how each coefficient evolves as regularization strength changes.
    Shows which features enter the model first (most important) and which
    are substitutable with correlated partners.
    """
    data = df[feature_cols + [target_col]].dropna()
    y = (data[target_col] == positive_class).astype(int)
    X = data[feature_cols]

    # Standardize so coefficients are comparable
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    # Range of regularization strengths (C = 1/alpha, so small C = strong regularization)
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

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color by dominant factor (if FA has been run — otherwise use a default palette)
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

    for i, col in enumerate(feature_cols):
        label = METRIC_LABELS.get(col, col).split(':')[0]  # short label
        ax.plot(C_values, coefs[col], label=label, linewidth=1.5, color=colors[i])

    ax.set_xscale('log')
    ax.set_xlabel('C (inverse regularization strength) →  weaker penalty →')
    ax.set_ylabel('Coefficient value')
    ax.set_title(f'Regularization path (L1){" — " + title if title else ""}')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig('regularization_path.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Entry order: at which C does each feature first become nonzero? ---
    entry_order = {}
    for col in feature_cols:
        for j, C in enumerate(C_values):
            if abs(coefs[col][j]) > 1e-6:
                entry_order[col] = C
                break
        else:
            entry_order[col] = np.inf

    entry_df = (
        pd.DataFrame({
            'Metric': list(entry_order.keys()),
            'Label': [METRIC_LABELS.get(m, m) for m in entry_order.keys()],
            'Entry_C': list(entry_order.values())
        })
        .sort_values('Entry_C')
    )
    entry_df['Rank'] = range(1, len(entry_df) + 1)
    print("\nFeature entry order (first to enter = most robustly important):")
    print(entry_df.to_string(index=False))

    return coefs, entry_df


# ======================================================================
# PART 4: Repeated Stratified K-Fold with Coefficient Stability
# ======================================================================

def repeated_kfold_coefficients(
    df, feature_cols, target_col, positive_class,
    n_splits=10, n_repeats=5, penalty='elasticnet', l1_ratio=0.5
):
    """
    Train the model across repeated stratified k-fold splits.
    Returns coefficient distributions + performance distributions.
    """
    data = df[feature_cols + [target_col]].dropna()
    y = (data[target_col] == positive_class).astype(int)
    X = data[feature_cols]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    all_coefs = []
    all_intercepts = []
    all_aucs = []
    all_briers = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_scaled, y)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LogisticRegression(
            penalty=penalty, solver='saga', l1_ratio=l1_ratio,
            max_iter=5000, class_weight='balanced', C=1.0
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        all_coefs.append(model.coef_[0])
        all_intercepts.append(model.intercept_[0])
        all_aucs.append(roc_auc_score(y_test, y_prob))
        all_briers.append(brier_score_loss(y_test, y_prob))

    coefs_array = np.array(all_coefs)  # shape: (n_folds*n_repeats, n_features)

    # --- Coefficient summary ---
    coef_summary = pd.DataFrame({
        'Metric': feature_cols,
        'Label': [METRIC_LABELS.get(m, m) for m in feature_cols],
        'Mean_Coef': coefs_array.mean(axis=0),
        'Std_Coef': coefs_array.std(axis=0),
        'CI_lower': np.percentile(coefs_array, 2.5, axis=0),
        'CI_upper': np.percentile(coefs_array, 97.5, axis=0),
        'Pct_nonzero': (np.abs(coefs_array) > 1e-6).mean(axis=0) * 100,
        'Sign_stable': [
            'YES' if (coefs_array[:, i] > 0).all() or (coefs_array[:, i] < 0).all() else
            'mostly' if (coefs_array[:, i] > 0).mean() > 0.9 or (coefs_array[:, i] < 0).mean() > 0.9 else
            'NO — UNSTABLE'
            for i in range(len(feature_cols))
        ]
    })
    coef_summary['Abs_Mean'] = coef_summary['Mean_Coef'].abs()
    coef_summary = coef_summary.sort_values('Abs_Mean', ascending=False)

    print(f"\nCoefficient stability across {n_splits}×{n_repeats} = {n_splits*n_repeats} fits:")
    print(coef_summary.to_string(index=False))

    print(f"\nPerformance across folds:")
    print(f"  ROC-AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"  Brier:   {np.mean(all_briers):.4f} ± {np.std(all_briers):.4f}")

    # --- Figure: Coefficient distributions ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by absolute mean coefficient
    order = coef_summary['Metric'].values
    positions = range(len(order))

    bp = ax.boxplot(
        [coefs_array[:, feature_cols.index(m)] for m in order],
        positions=positions,
        vert=True, patch_artist=True,
        widths=0.6
    )

    for patch in bp['boxes']:
        patch.set_facecolor('#534AB7')
        patch.set_alpha(0.4)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([METRIC_LABELS.get(m, m).split(':')[0] for m in order], rotation=45, ha='right')
    ax.set_ylabel('Coefficient value')
    ax.set_title(f'Coefficient stability ({n_splits}×{n_repeats} repeated stratified k-fold)')
    plt.tight_layout()
    plt.savefig('coefficient_stability.png', dpi=150, bbox_inches='tight')
    plt.show()

    return coef_summary, coefs_array, all_aucs


# ======================================================================
# PART 5: Calibrated Probability Cascade
# ======================================================================

def calibrated_cascade_stage(
    df, feature_cols, target_col, positive_class,
    stage_name="", n_splits=5, calibration_method='isotonic'
):
    """
    Train a calibrated classifier for one stage of the cascade.
    Uses CalibratedClassifierCV with cross-validation internally,
    so the calibration is done on held-out folds.

    Returns:
    - calibrated model (can call .predict_proba)
    - performance metrics
    - reliability diagram data
    """
    data = df[feature_cols + [target_col]].dropna().copy()
    y = (data[target_col] == positive_class).astype(int)
    X = data[feature_cols]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    # Base model
    base_model = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        max_iter=5000, class_weight='balanced', C=1.0
    )

    # Calibrated wrapper — uses internal CV to calibrate
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=calibration_method,
        cv=n_splits
    )
    calibrated_model.fit(X_scaled, y)

    # Cross-validated predictions for evaluation
    # (separate from calibration — this gives us honest out-of-sample probabilities)
    # NOTE: cross_val_predict requires a strict partition (each sample in exactly
    # one test fold), so we use StratifiedKFold, not RepeatedStratifiedKFold.
    from sklearn.model_selection import StratifiedKFold
    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We need a fresh calibrated model for cross_val_predict
    fresh_calibrated = CalibratedClassifierCV(
        estimator=LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=5000, class_weight='balanced', C=1.0
        ),
        method=calibration_method,
        cv=3  # inner CV for calibration
    )

    y_prob_cv = cross_val_predict(fresh_calibrated, X_scaled, y, cv=cv_eval, method='predict_proba')[:, 1]

    # --- Reliability diagram ---
    prob_true, prob_pred = calibration_curve(y, y_prob_cv, n_bins=10, strategy='quantile')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration plot
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Perfectly calibrated')
    ax1.plot(prob_pred, prob_true, 'o-', color='#534AB7', linewidth=2, label='Model')
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Observed frequency')
    ax1.set_title(f'Reliability diagram — {stage_name}')
    ax1.legend()

    # Histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(y_prob_cv[y == 1], bins=30, alpha=0.5, color='#1D9E75', label=f'{positive_class}', density=True)
    ax2.hist(y_prob_cv[y == 0], bins=30, alpha=0.5, color='#D85A30', label=f'not {positive_class}', density=True)
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Score distributions — {stage_name}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'calibration_{stage_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Performance ---
    auc = roc_auc_score(y, y_prob_cv)
    brier = brier_score_loss(y, y_prob_cv)

    y_pred_cv = (y_prob_cv >= 0.5).astype(int)
    report = classification_report(y, y_pred_cv, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Brier score: {brier:.4f}")
    print(f"  Classification report (at 0.5 threshold):")
    print(classification_report(y, y_pred_cv))

    # ECE (Expected Calibration Error)
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  (Lower = better calibrated. <0.05 is good, <0.02 is excellent)")

    return {
        'model': calibrated_model,
        'scaler': scaler,
        'auc': auc,
        'brier': brier,
        'ece': ece,
        'report': report,
        'y_prob_cv': y_prob_cv,
        'y_true': y,
        'calibration_curve': (prob_true, prob_pred)
    }


# ======================================================================
# PART 6: Full Pipeline — Run Everything
# ======================================================================

def run_full_pipeline(all_cities, blocks_labeled):
    """
    Execute the complete analysis pipeline.
    Expects all_cities and blocks_labeled to be GeoDataFrames
    with _std columns already computed.
    """

    print("=" * 70)
    print("  PART 1: PCA — Dimensionality Analysis")
    print("=" * 70)

    # Run on full dataset (all 102 cities)
    pca, scaler, loadings, n_retain = run_pca_analysis(
        all_cities, title_suffix=" (all cities)"
    )

    # Also run on validation blocks only (to check consistency)
    pca_val, _, loadings_val, n_retain_val = run_pca_analysis(
        blocks_labeled, title_suffix=" (validation blocks)"
    )

    print("\n" + "=" * 70)
    print("  PART 2: Factor Analysis — Latent Constructs")
    print("=" * 70)

    # Use the number of factors suggested by parallel analysis
    # (you can override this based on the scree plot)
    n_factors = n_retain
    print(f"\nUsing {n_factors} factors (from parallel analysis)")

    fa, fa_loadings, communalities = run_factor_analysis(
        all_cities, n_factors=n_factors, title_suffix=" (all cities)"
    )

    print("\n" + "=" * 70)
    print("  PART 3: Regularization Path — Feature Importance")
    print("=" * 70)

    # Subdivision vs irregular (core model)
    df_res = blocks_labeled[blocks_labeled['residential'] == 'residential'].copy()

    coefs_path, entry_df = plot_regularization_path(
        df=df_res,
        feature_cols=metrics_std,
        target_col='subdivisions_settlements',
        positive_class='subdivision',
        title='Subdivision vs Irregular'
    )

    print("\n" + "=" * 70)
    print("  PART 4: Coefficient Stability — Repeated K-Fold")
    print("=" * 70)

    coef_summary, coefs_array, aucs = repeated_kfold_coefficients(
        df=df_res,
        feature_cols=metrics_std,
        target_col='subdivisions_settlements',
        positive_class='subdivision'
    )

    print("\n" + "=" * 70)
    print("  PART 5: Calibrated Probability Cascade")
    print("=" * 70)

    # Stage 1: Built vs Open
    print("\n--- Stage 1: Built vs Open Space ---")
    stage1 = calibrated_cascade_stage(
        df=blocks_labeled,
        feature_cols=['m3_std', 'm4_std', 'm5_std', 'm7_std', 'm10_std', 'm11_std'],
        target_col='built_vs_open',
        positive_class='built_up',
        stage_name='Stage 1: Built vs Open'
    )

    # Stage 2: Residential vs Non-residential (among built-up)
    print("\n--- Stage 2: Residential vs Non-residential ---")
    df_built = blocks_labeled[blocks_labeled['built_vs_open'] == 'built_up'].copy()
    stage2 = calibrated_cascade_stage(
        df=df_built,
        feature_cols=metrics_std,
        target_col='residential',
        positive_class='residential',
        stage_name='Stage 2: Residential vs Non-res'
    )

    # Stage 3: Subdivision vs Irregular (the CORE model)
    print("\n--- Stage 3: Subdivision vs Irregular ---")
    stage3 = calibrated_cascade_stage(
        df=df_res,
        feature_cols=metrics_std,
        target_col='subdivisions_settlements',
        positive_class='subdivision',
        stage_name='Stage 3: Subdivision vs Irregular'
    )

    # Stage 4: Formal vs Informal (among subdivisions)
    print("\n--- Stage 4: Formal vs Informal Subdivisions ---")
    df_sub = df_res[df_res['subdivisions_settlements'] == 'subdivision'].copy()
    if df_sub['formal_informal_subdivisions'].nunique() >= 2:
        stage4 = calibrated_cascade_stage(
            df=df_sub,
            feature_cols=metrics_std,
            target_col='formal_informal_subdivisions',
            positive_class='formal_subdivisions',
            stage_name='Stage 4: Formal vs Informal'
        )
    else:
        print("  Skipping Stage 4 — insufficient classes")
        stage4 = None

    print("\n" + "=" * 70)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print(f"""
    PCA found {n_retain} meaningful dimensions in your 13 metrics.
    Factor analysis reveals the latent constructs — check the loadings
    heatmap to name them (e.g., 'road network quality', 'building pattern').

    Key results for the regularity index:
    - Stage 3 ROC-AUC (calibrated): {stage3['auc']:.4f}
    - Stage 3 ECE (calibration error): {stage3['ece']:.4f}
    - Stage 3 Brier score: {stage3['brier']:.4f}

    Coefficient stability (from repeated k-fold):
    """)
    # Show stable vs unstable coefficients
    stable = coef_summary[coef_summary['Sign_stable'] == 'YES']
    unstable = coef_summary[coef_summary['Sign_stable'] == 'NO — UNSTABLE']

    if len(stable) > 0:
        print("    STABLE coefficients (sign consistent across all folds):")
        for _, row in stable.iterrows():
            print(f"      {row['Label']:>45s}: {row['Mean_Coef']:+.3f} [{row['CI_lower']:+.3f}, {row['CI_upper']:+.3f}]")

    if len(unstable) > 0:
        print("\n    UNSTABLE coefficients (sign flips across folds — CAUTION):")
        for _, row in unstable.iterrows():
            print(f"      {row['Label']:>45s}: {row['Mean_Coef']:+.3f} [{row['CI_lower']:+.3f}, {row['CI_upper']:+.3f}]")

    return {
        'pca': pca, 'fa': fa,
        'coef_summary': coef_summary,
        'stage1': stage1, 'stage2': stage2,
        'stage3': stage3, 'stage4': stage4,
        'entry_order': entry_df
    }


# ======================================================================
# PART 7: Apply cascade to ALL blocks (scoring)
# ======================================================================

def score_all_blocks(all_cities, pipeline_results):
    """
    Apply the trained calibrated cascade to score every block
    in the full dataset. Adds columns:

    - p_built: P(built-up)
    - p_residential: P(residential | built)
    - p_subdivision: P(subdivision | residential)  ← THIS IS THE REGULARITY INDEX
    - p_formal: P(formal | subdivision)
    - regularity_index: = p_subdivision (for residential built-up blocks)
    - classification: hierarchical label
    """
    df = all_cities.copy()

    for stage_key, stage_name, feat_cols in [
        ('stage1', 'p_built', ['m3_std', 'm4_std', 'm5_std', 'm7_std', 'm10_std', 'm11_std']),
        ('stage2', 'p_residential', metrics_std),
        ('stage3', 'p_subdivision', metrics_std),
        ('stage4', 'p_formal', metrics_std),
    ]:
        stage = pipeline_results.get(stage_key)
        if stage is None:
            df[stage_name] = np.nan
            continue

        model = stage['model']
        scaler = stage['scaler']

        X = df[feat_cols].copy()
        valid_mask = X.notna().all(axis=1)
        X_valid = X.loc[valid_mask]

        if len(X_valid) > 0:
            X_scaled = scaler.transform(X_valid)
            probs = model.predict_proba(X_scaled)[:, 1]
            df.loc[valid_mask, stage_name] = probs
        else:
            df[stage_name] = np.nan

    # --- Hierarchical classification ---
    df['classification'] = 'unclassified'

    # Open space: P(built) < 0.5
    mask_open = df['p_built'] < 0.5
    df.loc[mask_open, 'classification'] = 'open_space'

    # Non-residential: built but P(residential) < 0.5
    mask_nonres = (~mask_open) & (df['p_residential'] < 0.5)
    df.loc[mask_nonres, 'classification'] = 'non_residential'

    # Irregular: residential but P(subdivision) < 0.5
    mask_irreg = (~mask_open) & (~mask_nonres) & (df['p_subdivision'] < 0.5)
    df.loc[mask_irreg, 'classification'] = 'irregular_settlement'

    # Subdivision
    mask_subdiv = (~mask_open) & (~mask_nonres) & (df['p_subdivision'] >= 0.5)
    df.loc[mask_subdiv, 'classification'] = 'subdivision'

    # Formal/informal subdivision
    if pipeline_results.get('stage4') is not None:
        mask_formal = mask_subdiv & (df['p_formal'] >= 0.5)
        mask_informal = mask_subdiv & (df['p_formal'] < 0.5)
        df.loc[mask_formal, 'classification'] = 'formal_subdivision'
        df.loc[mask_informal, 'classification'] = 'informal_subdivision'

    # --- Regularity index ---
    # Only defined for residential built-up blocks
    mask_residential_built = (~mask_open) & (~mask_nonres)
    df['regularity_index'] = np.nan
    df.loc[mask_residential_built, 'regularity_index'] = df.loc[mask_residential_built, 'p_subdivision']

    # --- Confidence measure: product of upstream probabilities ---
    # How "sure" the cascade is about reaching each stage
    df['cascade_confidence'] = np.nan
    df.loc[mask_residential_built, 'cascade_confidence'] = (
        df.loc[mask_residential_built, 'p_built'] *
        df.loc[mask_residential_built, 'p_residential']
    )

    print(f"\nScored {len(df):,} blocks")
    print(f"\nClassification distribution:")
    print(df['classification'].value_counts().to_string())
    print(f"\nRegularity index summary (residential built-up blocks only):")
    print(df['regularity_index'].describe().to_string())

    return df


# ======================================================================
# USAGE — paste into your notebook after Cell 10
# ======================================================================

if __name__ == "__main__":
    print("""
    ===================================================================
    To use this pipeline, paste the following into your notebook
    after Cell 10 (where all_cities and blocks_labeled exist):
    ===================================================================

    from pca_fa_pipeline import *

    # Run the full analysis
    results = run_full_pipeline(all_cities, blocks_labeled)

    # Score all blocks
    scored = score_all_blocks(all_cities, results)

    # Export
    scored.to_parquet('scored_blocks.geoparquet', index=False)
    """)
