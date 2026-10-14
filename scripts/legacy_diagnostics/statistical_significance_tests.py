"""
Statistical significance tests: Is the Fusion model truly better than the bookmaker?

Tests performed:
1. Paired t-test on per-match log-loss differences
2. Bootstrap confidence intervals for LogLoss, AUC, Brier differences
3. Diebold-Mariano test (for forecast comparison)
4. Wilcoxon signed-rank test (non-parametric paired test)
5. McNemar's test (for accuracy/disagreement)
6. Permutation test (exact test for LogLoss difference)

All tests on 2025+ data with valid odds (same 2320-match subset).
"""
import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

np.random.seed(42)

# ============================================
# Load and prepare data (same as compare_all_models_odds.py)
# ============================================
print("Loading data...")
odds = pd.read_csv('data/odds.csv')
baseline = pd.read_csv('data/golgg_y_predicts.csv')

with open('data/fusion_predictions_all.json', 'r') as f:
    fusion_preds = json.load(f)

# Merge
merged = odds.merge(baseline, on='golgg_match_id', how='inner', suffixes=('_odds', '_base'))

# Team alignment
merged['teams_match'] = (merged['golgg_team1'] == merged['team1_name']) & (merged['golgg_team2'] == merged['team2_name'])
merged['teams_swapped'] = (merged['golgg_team1'] == merged['team2_name']) & (merged['golgg_team2'] == merged['team1_name'])
merged['teams_neither'] = ~merged['teams_match'] & ~merged['teams_swapped']

# Valid odds
merged = merged[merged['avg_odds_home'].notna() & merged['avg_odds_away'].notna()].copy()

# Bookmaker probabilities (aligned to baseline team1)
merged['bm_p_team1'] = np.where(
    merged['teams_match'],
    1.0 / merged['avg_odds_home'],
    np.where(merged['teams_swapped'], 1.0 / merged['avg_odds_away'], np.nan)
)
merged['bm_p_team2'] = np.where(
    merged['teams_match'],
    1.0 / merged['avg_odds_away'],
    np.where(merged['teams_swapped'], 1.0 / merged['avg_odds_home'], np.nan)
)

merged = merged[merged['bm_p_team1'].notna()].copy()
merged['bm_p_team1_norm'] = merged['bm_p_team1'] / (merged['bm_p_team1'] + merged['bm_p_team2'])

# Odds aligned
merged['odds_team1'] = np.where(
    merged['teams_match'],
    merged['avg_odds_home'],
    np.where(merged['teams_swapped'], merged['avg_odds_away'], np.nan)
)
merged['odds_team2'] = np.where(
    merged['teams_match'],
    merged['avg_odds_away'],
    np.where(merged['teams_swapped'], merged['avg_odds_home'], np.nan)
)

# Fusion predictions
merged['match_id_str'] = merged['golgg_match_id'].astype(str)
merged['fusion_v2'] = merged['match_id_str'].map(lambda x: fusion_preds.get(x, {}).get('fusion_v2', np.nan))
merged['fusion_v2_sym'] = merged['match_id_str'].map(lambda x: fusion_preds.get(x, {}).get('fusion_v2_sym', np.nan))
merged['fusion_v2_archsym'] = merged['match_id_str'].map(lambda x: fusion_preds.get(x, {}).get('fusion_v2_archsym', np.nan))

# Filter to 2025+ test set
merged['date_base'] = pd.to_datetime(merged['date'])
test = merged[merged['date_base'] >= '2025-01-01'].copy()

# Common subset where all models have predictions
common_mask = test['fusion_v2'].notna() & test['fusion_v2_sym'].notna() & test['fusion_v2_archsym'].notna()
test = test[common_mask].copy()
test = test.reset_index(drop=True)

print(f"Test set: {len(test)} matches (2025+, with valid odds and all model predictions)")

y_true = test['y_true'].values.astype(float)
bm_probs = test['bm_p_team1_norm'].values
baseline_probs = test['player_elo'].values
fusion_probs = test['fusion_v2'].values
fusion_sym_probs = test['fusion_v2_sym'].values
fusion_arch_probs = test['fusion_v2_archsym'].values

# Clip probabilities for numerical stability
EPS = 1e-7
bm_probs_c = np.clip(bm_probs, EPS, 1 - EPS)
baseline_probs_c = np.clip(baseline_probs, EPS, 1 - EPS)
fusion_probs_c = np.clip(fusion_probs, EPS, 1 - EPS)
fusion_sym_probs_c = np.clip(fusion_sym_probs, EPS, 1 - EPS)
fusion_arch_probs_c = np.clip(fusion_arch_probs, EPS, 1 - EPS)

# ============================================
# Helper: per-sample log-loss
# ============================================
def per_sample_logloss(y_true, y_pred):
    """Compute log-loss for each sample individually."""
    y_pred_c = np.clip(y_pred, EPS, 1 - EPS)
    return -y_true * np.log(y_pred_c) - (1 - y_true) * np.log(1 - y_pred_c)

def per_sample_brier(y_true, y_pred):
    """Compute Brier score for each sample."""
    return (y_pred - y_true) ** 2

# ============================================
# 1. PAIRED T-TEST ON PER-MATCH LOG-LOSS
# ============================================
print("\n" + "="*70)
print("1. PAIRED T-TEST ON PER-MATCH LOG-LOSS")
print("="*70)

comparisons = [
    ("Fusion v2 vs Bookmaker", fusion_probs_c, bm_probs_c),
    ("Fusion v2 vs Baseline", fusion_probs_c, baseline_probs_c),
    ("Fusion v2+SymAug vs Bookmaker", fusion_sym_probs_c, bm_probs_c),
    ("Fusion v2+ArchSym vs Bookmaker", fusion_arch_probs_c, bm_probs_c),
    ("Fusion v2 vs Fusion+SymAug", fusion_probs_c, fusion_sym_probs_c),
]

for name, probs_a, probs_b in comparisons:
    ll_a = per_sample_logloss(y_true, probs_a)
    ll_b = per_sample_logloss(y_true, probs_b)
    diff = ll_a - ll_b  # negative = model A is better (lower log-loss)
    
    t_stat, p_val = stats.ttest_rel(ll_a, ll_b)
    mean_diff = diff.mean()
    std_diff = diff.std()
    ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=std_diff/np.sqrt(len(diff)))
    
    print(f"\n{name}:")
    print(f"  Mean LL diff (A-B): {mean_diff:.6f} {'(A better)' if mean_diff < 0 else '(B better)'}")
    print(f"  Std of diff:        {std_diff:.6f}")
    print(f"  t-statistic:        {t_stat:.4f}")
    print(f"  p-value (two-sided): {p_val:.6f}")
    print(f"  p-value (one-sided): {p_val/2:.6f}")
    print(f"  95% CI of diff:     [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
    print(f"  Significant (α=0.05, one-sided): {'YES ✓' if p_val/2 < 0.05 and mean_diff < 0 else 'NO ✗'}")

# ============================================
# 2. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================
print("\n" + "="*70)
print("2. BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
print("="*70)

N_BOOTSTRAP = 10000
n = len(y_true)

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap a metric and return distribution."""
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], np.clip(y_pred[idx], EPS, 1-EPS))
            scores.append(score)
        except:
            pass
    return np.array(scores)

def bootstrap_diff(y_true, y_pred_a, y_pred_b, metric_fn, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap the difference in a metric (A - B)."""
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        try:
            sa = metric_fn(y_true[idx], np.clip(y_pred_a[idx], EPS, 1-EPS))
            sb = metric_fn(y_true[idx], np.clip(y_pred_b[idx], EPS, 1-EPS))
            diffs.append(sa - sb)
        except:
            pass
    return np.array(diffs)

print("\n--- LogLoss Bootstrap ---")
for name, probs in [("Bookmaker", bm_probs_c), ("Baseline", baseline_probs_c), 
                     ("Fusion v2", fusion_probs_c), ("Fusion+SymAug", fusion_sym_probs_c),
                     ("Fusion+ArchSym", fusion_arch_probs_c)]:
    boot_scores = bootstrap_metric(y_true, probs, log_loss)
    print(f"  {name:20s}: {boot_scores.mean():.4f} [{np.percentile(boot_scores, 2.5):.4f}, {np.percentile(boot_scores, 97.5):.4f}]")

print("\n--- LogLoss DIFFERENCE Bootstrap (Model - Bookmaker) ---")
for name, probs in [("Fusion v2", fusion_probs_c), ("Fusion+SymAug", fusion_sym_probs_c),
                     ("Fusion+ArchSym", fusion_arch_probs_c), ("Baseline", baseline_probs_c)]:
    boot_diff = bootstrap_diff(y_true, probs, bm_probs_c, log_loss)
    ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
    p_one_sided = (boot_diff > 0).mean()  # P(diff > 0) = P(bookmaker better)
    print(f"  {name:20s} - BM: {boot_diff.mean():.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  P(BM better)={p_one_sided:.4f}  Sig={'YES' if ci_hi < 0 else 'NO'}")

print("\n--- AUC Bootstrap ---")
for name, probs in [("Bookmaker", bm_probs_c), ("Baseline", baseline_probs_c), 
                     ("Fusion v2", fusion_probs_c), ("Fusion+SymAug", fusion_sym_probs_c),
                     ("Fusion+ArchSym", fusion_arch_probs_c)]:
    boot_scores = bootstrap_metric(y_true, probs, roc_auc_score)
    print(f"  {name:20s}: {boot_scores.mean():.4f} [{np.percentile(boot_scores, 2.5):.4f}, {np.percentile(boot_scores, 97.5):.4f}]")

print("\n--- AUC DIFFERENCE Bootstrap (Model - Bookmaker) ---")
for name, probs in [("Fusion v2", fusion_probs_c), ("Fusion+SymAug", fusion_sym_probs_c),
                     ("Fusion+ArchSym", fusion_arch_probs_c), ("Baseline", baseline_probs_c)]:
    boot_diff = bootstrap_diff(y_true, probs, bm_probs_c, roc_auc_score)
    ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
    p_one_sided = (boot_diff < 0).mean()  # P(diff < 0) = P(bookmaker better AUC)
    print(f"  {name:20s} - BM: {boot_diff.mean():.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  P(BM better)={p_one_sided:.4f}  Sig={'YES' if ci_lo > 0 else 'NO'}")

print("\n--- Brier Score Bootstrap ---")
for name, probs in [("Bookmaker", bm_probs_c), ("Baseline", baseline_probs_c), 
                     ("Fusion v2", fusion_probs_c), ("Fusion+SymAug", fusion_sym_probs_c),
                     ("Fusion+ArchSym", fusion_arch_probs_c)]:
    boot_scores = bootstrap_metric(y_true, probs, brier_score_loss)
    print(f"  {name:20s}: {boot_scores.mean():.4f} [{np.percentile(boot_scores, 2.5):.4f}, {np.percentile(boot_scores, 97.5):.4f}]")

# ============================================
# 3. DIEBOLD-MARIANO TEST
# ============================================
print("\n" + "="*70)
print("3. DIEBOLD-MARIANO TEST (forecast accuracy comparison)")
print("="*70)

def diebold_mariano_test(errors_a, errors_b, h=1, alternative='two-sided'):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    H0: Both forecasts have the same accuracy (E[d_t] = 0)
    H1: Forecasts have different accuracy
    
    errors_a, errors_b: per-observation loss values (e.g., log-loss per match)
    h: forecast horizon (1 for one-step-ahead)
    """
    d = errors_a - errors_b
    d_mean = d.mean()
    
    # Newey-West HAC variance estimator
    n = len(d)
    gamma_0 = np.var(d, ddof=0)
    
    # Compute autocovariances up to lag h-1
    gamma_sum = 0.0
    for lag in range(1, h):
        if lag < n:
            gamma_lag = np.mean((d[:n-lag] - d_mean) * (d[n-lag:] - d_mean))
            gamma_sum += 2 * gamma_lag
    
    var_d = gamma_0 + gamma_sum
    se_d = np.sqrt(var_d / n)
    
    dm_stat = d_mean / se_d
    
    # Approximate p-value using normal distribution
    if alternative == 'two-sided':
        p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == 'less':  # H1: E[d] < 0 (A is better)
        p_val = stats.norm.cdf(dm_stat)
    elif alternative == 'greater':
        p_val = 1 - stats.norm.cdf(dm_stat)
    
    return dm_stat, p_val

for name, probs_a, probs_b in [
    ("Fusion v2 vs Bookmaker", fusion_probs_c, bm_probs_c),
    ("Fusion v2 vs Baseline", fusion_probs_c, baseline_probs_c),
    ("Fusion v2+SymAug vs Bookmaker", fusion_sym_probs_c, bm_probs_c),
    ("Fusion v2+ArchSym vs Bookmaker", fusion_arch_probs_c, bm_probs_c),
    ("Fusion v2 vs Fusion+SymAug", fusion_probs_c, fusion_sym_probs_c),
]:
    ll_a = per_sample_logloss(y_true, probs_a)
    ll_b = per_sample_logloss(y_true, probs_b)
    
    dm_stat, p_val_two = diebold_mariano_test(ll_a, ll_b, h=1, alternative='two-sided')
    _, p_val_less = diebold_mariano_test(ll_a, ll_b, h=1, alternative='less')
    
    print(f"\n{name}:")
    print(f"  DM statistic:       {dm_stat:.4f}")
    print(f"  p-value (two-sided): {p_val_two:.6f}")
    print(f"  p-value (A < B):    {p_val_less:.6f}")
    print(f"  Significant (α=0.05, A better): {'YES ✓' if p_val_less < 0.05 else 'NO ✗'}")

# ============================================
# 4. WILCOXON SIGNED-RANK TEST
# ============================================
print("\n" + "="*70)
print("4. WILCOXON SIGNED-RANK TEST (non-parametric paired test)")
print("="*70)

for name, probs_a, probs_b in [
    ("Fusion v2 vs Bookmaker", fusion_probs_c, bm_probs_c),
    ("Fusion v2 vs Baseline", fusion_probs_c, baseline_probs_c),
    ("Fusion v2+SymAug vs Bookmaker", fusion_sym_probs_c, bm_probs_c),
    ("Fusion v2+ArchSym vs Bookmaker", fusion_arch_probs_c, bm_probs_c),
    ("Fusion v2 vs Fusion+SymAug", fusion_probs_c, fusion_sym_probs_c),
]:
    ll_a = per_sample_logloss(y_true, probs_a)
    ll_b = per_sample_logloss(y_true, probs_b)
    diff = ll_a - ll_b
    
    # Remove zeros (exact ties)
    diff_nonzero = diff[diff != 0]
    
    stat, p_val_two = stats.wilcoxon(diff_nonzero, alternative='two-sided')
    stat_less, p_val_less = stats.wilcoxon(diff_nonzero, alternative='less')
    
    n_positive = (diff < 0).sum()  # A better
    n_negative = (diff > 0).sum()  # B better
    n_ties = (diff == 0).sum()
    
    print(f"\n{name}:")
    print(f"  A better: {n_positive}, B better: {n_negative}, Ties: {n_ties}")
    print(f"  W statistic:        {stat:.1f}")
    print(f"  p-value (two-sided): {p_val_two:.6f}")
    print(f"  p-value (A < B):    {p_val_less:.6f}")
    print(f"  Significant (α=0.05, A better): {'YES ✓' if p_val_less < 0.05 else 'NO ✗'}")

# ============================================
# 5. McNEMAR'S TEST (accuracy comparison)
# ============================================
print("\n" + "="*70)
print("5. McNEMAR'S TEST (accuracy / disagreement)")
print("="*70)

for name, probs_a, probs_b in [
    ("Fusion v2 vs Bookmaker", fusion_probs_c, bm_probs_c),
    ("Fusion v2 vs Baseline", fusion_probs_c, baseline_probs_c),
    ("Fusion v2+SymAug vs Bookmaker", fusion_sym_probs_c, bm_probs_c),
    ("Fusion v2+ArchSym vs Bookmaker", fusion_arch_probs_c, bm_probs_c),
]:
    pred_a = (probs_a > 0.5).astype(int)
    pred_b = (probs_b > 0.5).astype(int)
    y = y_true.astype(int)
    
    correct_a = (pred_a == y)
    correct_b = (pred_b == y)
    
    # Contingency table
    n_both_correct = (correct_a & correct_b).sum()
    n_a_only = (correct_a & ~correct_b).sum()  # A correct, B wrong
    n_b_only = (~correct_a & correct_b).sum()  # B correct, A wrong
    n_both_wrong = (~correct_a & ~correct_b).sum()
    
    # McNemar's test uses discordant pairs
    # H0: P(A correct, B wrong) = P(B correct, A wrong)
    b_mcnemar = n_a_only  # A correct, B wrong
    c_mcnemar = n_b_only  # B correct, A wrong
    
    if b_mcnemar + c_mcnemar > 0:
        # With continuity correction
        mcnemar_stat = (abs(b_mcnemar - c_mcnemar) - 1) ** 2 / (b_mcnemar + c_mcnemar)
        p_val = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        
        # Exact binomial test (more appropriate for small counts)
        p_val_exact = 2 * stats.binom.cdf(min(b_mcnemar, c_mcnemar), b_mcnemar + c_mcnemar, 0.5)
    else:
        mcnemar_stat = 0
        p_val = 1.0
        p_val_exact = 1.0
    
    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    
    print(f"\n{name}:")
    print(f"  Acc A: {acc_a:.4f}, Acc B: {acc_b:.4f}")
    print(f"  Both correct: {n_both_correct}, A only: {n_a_only}, B only: {n_b_only}, Both wrong: {n_both_wrong}")
    print(f"  McNemar χ²:         {mcnemar_stat:.4f}")
    print(f"  p-value (asymptotic): {p_val:.6f}")
    print(f"  p-value (exact):     {p_val_exact:.6f}")
    print(f"  Significant (α=0.05): {'YES ✓' if p_val_exact < 0.05 else 'NO ✗'}")

# ============================================
# 6. PERMUTATION TEST (exact test for LogLoss difference)
# ============================================
print("\n" + "="*70)
print("6. PERMUTATION TEST FOR LogLoss DIFFERENCE (10,000 permutations)")
print("="*70)

N_PERM = 10000

for name, probs_a, probs_b in [
    ("Fusion v2 vs Bookmaker", fusion_probs_c, bm_probs_c),
    ("Fusion v2 vs Baseline", fusion_probs_c, baseline_probs_c),
    ("Fusion v2+SymAug vs Bookmaker", fusion_sym_probs_c, bm_probs_c),
    ("Fusion v2+ArchSym vs Bookmaker", fusion_arch_probs_c, bm_probs_c),
]:
    ll_a = per_sample_logloss(y_true, probs_a)
    ll_b = per_sample_logloss(y_true, probs_b)
    
    observed_diff = ll_a.mean() - ll_b.mean()  # negative = A better
    
    # Permutation: randomly swap which model's loss is assigned to which
    perm_diffs = []
    for _ in range(N_PERM):
        swap = np.random.randint(0, 2, size=n).astype(bool)
        ll_perm_a = np.where(swap, ll_b, ll_a)
        ll_perm_b = np.where(swap, ll_a, ll_b)
        perm_diffs.append(ll_perm_a.mean() - ll_perm_b.mean())
    
    perm_diffs = np.array(perm_diffs)
    
    # One-sided p-value: P(perm_diff <= observed_diff) 
    # If A is better, observed_diff < 0, so we count how many perms are even more extreme
    p_val_one = (perm_diffs <= observed_diff).mean()
    p_val_two = 2 * min(p_val_one, 1 - p_val_one)
    
    print(f"\n{name}:")
    print(f"  Observed diff (A-B): {observed_diff:.6f}")
    print(f"  Permutation mean:    {perm_diffs.mean():.6f}")
    print(f"  Permutation std:     {perm_diffs.std():.6f}")
    print(f"  p-value (one-sided): {p_val_one:.6f}")
    print(f"  p-value (two-sided): {p_val_two:.6f}")
    print(f"  Significant (α=0.05, A better): {'YES ✓' if p_val_one < 0.05 else 'NO ✗'}")

# ============================================
# 7. ADDITIONAL: Bootstrap on ROI (betting simulation)
# ============================================
print("\n" + "="*70)
print("7. BOOTSTRAP ON BETTING ROI (Fusion v2 vs Bookmaker, edge > 0)")
print("="*70)

def compute_roi(y_true, model_probs, odds_t1, odds_t2, min_edge=0.0):
    """Compute ROI for a given model and minimum edge."""
    profit = 0.0
    bets = 0
    for i in range(len(y_true)):
        p_model = model_probs[i]
        o1 = odds_t1[i]
        o2 = odds_t2[i]
        if np.isnan(o1) or np.isnan(o2) or o1 <= 1 or o2 <= 1:
            continue
        imp1 = 1.0 / o1
        imp2 = 1.0 / o2
        edge1 = p_model - imp1
        edge2 = (1 - p_model) - imp2
        if edge1 > min_edge:
            bets += 1
            if y_true[i] == 1:
                profit += (o1 - 1)
            else:
                profit -= 1
        elif edge2 > min_edge:
            bets += 1
            if y_true[i] == 0:
                profit += (o2 - 1)
            else:
                profit -= 1
    roi = profit / bets * 100 if bets > 0 else 0
    return roi, bets, profit

odds_t1 = test['odds_team1'].values
odds_t2 = test['odds_team2'].values

for min_edge in [0.0, 0.05, 0.10]:
    print(f"\n--- Edge > {min_edge:.2f} ---")
    
    # Bootstrap ROI for Fusion v2
    roi_boots = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n, size=n)
        roi, _, _ = compute_roi(y_true[idx], fusion_probs[idx], odds_t1[idx], odds_t2[idx], min_edge)
        roi_boots.append(roi)
    roi_boots = np.array(roi_boots)
    
    # Bootstrap ROI for Baseline
    roi_boots_base = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n, size=n)
        roi, _, _ = compute_roi(y_true[idx], baseline_probs[idx], odds_t1[idx], odds_t2[idx], min_edge)
        roi_boots_base.append(roi)
    roi_boots_base = np.array(roi_boots_base)
    
    roi_obs, bets_obs, profit_obs = compute_roi(y_true, fusion_probs, odds_t1, odds_t2, min_edge)
    roi_obs_base, bets_obs_base, profit_obs_base = compute_roi(y_true, baseline_probs, odds_t1, odds_t2, min_edge)
    
    print(f"  Fusion v2:  ROI={roi_obs:.1f}% (bets={bets_obs}, profit={profit_obs:.1f})")
    print(f"    Bootstrap 95% CI: [{np.percentile(roi_boots, 2.5):.1f}%, {np.percentile(roi_boots, 97.5):.1f}%]")
    print(f"    P(ROI > 0): {(roi_boots > 0).mean():.4f}")
    print(f"  Baseline:   ROI={roi_obs_base:.1f}% (bets={bets_obs_base}, profit={profit_obs_base:.1f})")
    print(f"    Bootstrap 95% CI: [{np.percentile(roi_boots_base, 2.5):.1f}%, {np.percentile(roi_boots_base, 97.5):.1f}%]")
    print(f"    P(ROI > 0): {(roi_boots_base > 0).mean():.4f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("SUMMARY: Is Fusion v2 significantly better than Bookmaker?")
print("="*70)

ll_fusion = per_sample_logloss(y_true, fusion_probs_c)
ll_bm = per_sample_logloss(y_true, bm_probs_c)
diff = ll_fusion - ll_bm

print(f"\nLogLoss difference (Fusion - Bookmaker): {diff.mean():.6f}")
print(f"  Negative = Fusion is better")
print(f"  Fusion wins on {((ll_fusion < ll_bm)).sum()}/{n} matches ({(ll_fusion < ll_bm).mean()*100:.1f}%)")

# Collect all p-values
print(f"\n{'Test':<35} {'p-value (one-sided)':<20} {'Significant?':<15}")
print("-" * 70)

# Paired t-test
_, p_t = stats.ttest_rel(ll_fusion, ll_bm)
print(f"{'Paired t-test':<35} {p_t/2:<20.6f} {'YES ✓' if p_t/2 < 0.05 else 'NO ✗':<15}")

# Wilcoxon
diff_nz = diff[diff != 0]
_, p_w = stats.wilcoxon(diff_nz, alternative='less')
print(f"{'Wilcoxon signed-rank':<35} {p_w:<20.6f} {'YES ✓' if p_w < 0.05 else 'NO ✗':<15}")

# DM test
_, p_dm = diebold_mariano_test(ll_fusion, ll_bm, h=1, alternative='less')
print(f"{'Diebold-Mariano':<35} {p_dm:<20.6f} {'YES ✓' if p_dm < 0.05 else 'NO ✗':<15}")

# Permutation test
perm_diffs = []
for _ in range(N_PERM):
    swap = np.random.randint(0, 2, size=n).astype(bool)
    ll_pa = np.where(swap, ll_bm, ll_fusion)
    ll_pb = np.where(swap, ll_fusion, ll_bm)
    perm_diffs.append(ll_pa.mean() - ll_pb.mean())
perm_diffs = np.array(perm_diffs)
p_perm = (perm_diffs <= diff.mean()).mean()
print(f"{'Permutation test':<35} {p_perm:<20.6f} {'YES ✓' if p_perm < 0.05 else 'NO ✗':<15}")

# Bootstrap
boot_diff_ll = bootstrap_diff(y_true, fusion_probs_c, bm_probs_c, log_loss)
p_boot = (boot_diff_ll > 0).mean()  # P(bookmaker better)
ci_lo, ci_hi = np.percentile(boot_diff_ll, [2.5, 97.5])
print(f"{'Bootstrap (LL diff CI)':<35} CI=[{ci_lo:.4f}, {ci_hi:.4f}]  {'YES ✓' if ci_hi < 0 else 'NO ✗'}")

print(f"\nConclusion: Fusion v2 is {'SIGNIFICANTLY' if p_t/2 < 0.05 and p_w < 0.05 else 'NOT significantly'} better than Bookmaker (α=0.05)")

# Save results
results = {
    'n_matches': int(n),
    'logloss_fusion': float(log_loss(y_true, fusion_probs_c)),
    'logloss_bookmaker': float(log_loss(y_true, bm_probs_c)),
    'logloss_diff': float(diff.mean()),
    'paired_t_test_p': float(p_t/2),
    'wilcoxon_p': float(p_w),
    'diebold_mariano_p': float(p_dm),
    'permutation_p': float(p_perm),
    'bootstrap_ll_diff_ci': [float(ci_lo), float(ci_hi)],
    'significant_at_005': bool(p_t/2 < 0.05 and p_w < 0.05),
}

with open('data/statistical_significance_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to data/statistical_significance_results.json")
