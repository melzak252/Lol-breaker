"""
Comprehensive statistical tests for model competitiveness and equal predictive ability:
1. Non-Inferiority Test (Test niegorszości w ramach TOST)
2. Diebold-Mariano (DM) & Harvey-Leybourne-Newbold (HLN) Tests (Equal Predictive Ability)
3. Model Confidence Set (MCS, Hansen, Lunde, Nason 2011)
4. Forecast Encompassing Tests (Harvey, Leybourne, Newbold 1998)
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

EPS = 1e-15

def binary_log_loss_vector(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

def brier_score_vector(y_true, y_pred):
    return (y_pred - y_true) ** 2

def load_canonical_data():
    path = "data/08_reporting/thesis_lr_corrections_20260924/paired_predictions.parquet"
    df = pd.read_parquet(path)
    # Market common sample
    market_common = df[df[["market_open", "market_close"]].notna().all(axis=1)].copy()
    return df, market_common

def run_non_inferiority_tests(data, candidate_col, reference_col, candidate_name, reference_name, sample_name,
                               margins=[0.001, 0.002, 0.003, 0.004, 0.005, 0.010], n_resamples=10000, seed=42):
    """
    Non-Inferiority Test:
    H0: mu_candidate - mu_reference >= +epsilon (candidate is inferior by >= epsilon)
    H1: mu_candidate - mu_reference < +epsilon (candidate is non-inferior / competitive)
    
    Using monthly block bootstrap to respect calendar clustering.
    """
    rng = np.random.default_rng(seed)
    y = data["y_true"].to_numpy()
    p_cand = data[candidate_col].to_numpy()
    p_ref = data[reference_col].to_numpy()
    
    loss_cand = binary_log_loss_vector(y, p_cand)
    loss_ref = binary_log_loss_vector(y, p_ref)
    loss_diff = loss_cand - loss_ref
    
    observed_delta = float(loss_diff.mean())
    
    # Month grouping
    dates = pd.to_datetime(data["date"])
    months = dates.dt.to_period("M")
    unique_months = months.unique()
    m_to_indices = {m: np.where(months == m)[0] for m in unique_months}
    n_months = len(unique_months)
    
    # Bootstrap
    boot_deltas = np.empty(n_resamples)
    for b in range(n_resamples):
        sampled_months = rng.choice(unique_months, size=n_months, replace=True)
        idx = np.concatenate([m_to_indices[m] for m in sampled_months])
        boot_deltas[b] = loss_diff[idx].mean()
        
    ci_90 = np.percentile(boot_deltas, [5.0, 95.0])
    ci_95 = np.percentile(boot_deltas, [2.5, 97.5])
    
    results = {
        "candidate": candidate_name,
        "reference": reference_name,
        "sample": sample_name,
        "N": len(data),
        "observed_delta": observed_delta,
        "ci_90": ci_90,
        "ci_95": ci_95,
        "margins": {}
    }
    
    for eps in margins:
        # P(H0) = fraction of bootstrap samples where delta >= +eps
        p_val = float((boot_deltas >= eps).mean())
        # Decision at alpha = 0.05
        is_non_inferior = p_val < 0.05
        results["margins"][eps] = {
            "p_value": p_val,
            "non_inferior": is_non_inferior
        }
    return results

def diebold_mariano_test(loss_cand, loss_ref, h=1):
    """
    Diebold-Mariano & Harvey-Leybourne-Newbold tests for equal predictive ability.
    H0: E[d] = 0 (Equal Predictive Ability)
    H1: E[d] != 0 (Two-sided)
    H1_less: E[d] < 0 (Candidate is better)
    """
    d = loss_cand - loss_ref
    n = len(d)
    d_bar = np.mean(d)
    
    # Autocovariances up to lag h-1 with Bartlett weights (Newey-West)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        if lag < n:
            weight = 1.0 - lag / h
            gamma_lag = np.mean((d[:n-lag] - d_bar) * (d[lag:] - d_bar))
            gamma_sum += 2.0 * weight * gamma_lag
            
    hac_var = (gamma_0 + gamma_sum) / n
    hac_se = np.sqrt(max(hac_var, 1e-12))
    dm_stat = d_bar / hac_se
    
    # HLN small-sample correction
    hln_corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if n > 0 else 1.0
    hln_stat = dm_stat * hln_corr
    
    p_two_sided = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    p_candidate_better = stats.norm.cdf(dm_stat)
    p_candidate_worse = 1.0 - stats.norm.cdf(dm_stat)
    
    p_hln_two_sided = 2.0 * (1.0 - stats.t.cdf(abs(hln_stat), df=n-1))
    p_hln_candidate_better = stats.t.cdf(hln_stat, df=n-1)
    
    return {
        "mean_diff": float(d_bar),
        "hac_se": float(hac_se),
        "dm_stat": float(dm_stat),
        "p_two_sided": float(p_two_sided),
        "p_candidate_better": float(p_candidate_better),
        "hln_stat": float(hln_stat),
        "p_hln_two_sided": float(p_hln_two_sided),
        "p_hln_better": float(p_hln_candidate_better)
    }

def model_confidence_set(loss_matrix, model_names, alpha=0.05, n_resamples=2000, seed=42):
    """
    Model Confidence Set (MCS, Hansen, Lunde, Nason 2011).
    Sequential elimination procedure to identify the superior set of models.
    """
    rng = np.random.default_rng(seed)
    n, m = loss_matrix.shape
    surviving_indices = list(range(m))
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    
    p_values = {}
    eliminated_order = []
    
    while len(surviving_indices) > 1:
        current_m = len(surviving_indices)
        sub_losses = loss_matrix[:, surviving_indices]
        
        # d_ij = L_i - L_j
        # d_i. = 1/(m-1) sum_{j != i} (L_i - L_j)
        d_bar = np.zeros(current_m)
        d_se = np.zeros(current_m)
        
        # Mean across rows
        means = sub_losses.mean(axis=0) # shape (current_m,)
        grand_mean = means.mean()
        
        # d_bar_i = mean(L_i) - 1/current_m * sum_j mean(L_j)
        for i in range(current_m):
            d_i = sub_losses[:, i] - sub_losses.mean(axis=1)
            d_bar[i] = d_i.mean()
            # Bootstrap SE for d_i
            boot_means = sub_losses[boot_indices, i].mean(axis=1) - sub_losses[boot_indices, :].mean(axis=2).mean(axis=1)
            d_se[i] = np.std(boot_means, ddof=1)
            if d_se[i] < 1e-12:
                d_se[i] = 1e-12
                
        # T_max statistic: max_i (d_bar_i / SE(d_bar_i))
        t_stats = d_bar / d_se
        t_max = np.max(t_stats)
        worst_i_in_sub = np.argmax(t_stats)
        worst_model_idx = surviving_indices[worst_i_in_sub]
        
        # Bootstrap distribution of T_max under H0
        boot_t_max = np.zeros(n_resamples)
        for b in range(n_resamples):
            b_idx = boot_indices[b]
            b_losses = sub_losses[b_idx, :]
            b_means = b_losses.mean(axis=0) - means # centered under H0
            b_d_i = b_means - b_means.mean()
            b_t = b_d_i / d_se
            boot_t_max[b] = np.max(b_t)
            
        p_val_step = float((boot_t_max >= t_max).mean())
        p_values[worst_model_idx] = p_val_step
        
        if p_val_step < alpha:
            # Reject H0, eliminate worst model
            eliminated_order.append(worst_model_idx)
            surviving_indices.pop(worst_i_in_sub)
        else:
            # Cannot reject H0, stop elimination
            break
            
    # For all surviving models, set MCS p-value to max remaining
    for idx in surviving_indices:
        p_values[idx] = 1.0
        
    mcs_models = [model_names[i] for i in surviving_indices]
    all_results = []
    for i, name in enumerate(model_names):
        all_results.append({
            "model": name,
            "mean_loss": float(loss_matrix[:, i].mean()),
            "mcs_p_value": p_values[i],
            "in_mcs_95": p_values[i] >= 0.05,
            "in_mcs_90": p_values[i] >= 0.10
        })
    all_results.sort(key=lambda x: x["mean_loss"])
    return all_results

def forecast_encompassing_test(y, p_cand, p_ref, candidate_name, reference_name):
    """
    Forecast Encompassing Test (Harvey, Leybourne, Newbold 1998):
    y = beta_0 + beta_1 * p_ref + beta_2 * p_cand + eps
    
    H0_1: beta_2 = 0 (Reference encompasses Candidate; Candidate adds no info)
    H0_2: beta_1 = 0 (Candidate encompasses Reference; Reference adds no info)
    """
    # Robust Logit Encompassing Regression
    # We clip probabilities to avoid logit infinities
    p_c = np.clip(p_cand, 0.001, 0.999)
    p_r = np.clip(p_ref, 0.001, 0.999)
    logit_cand = np.log(p_c / (1.0 - p_c))
    logit_ref = np.log(p_r / (1.0 - p_r))
    X = np.column_stack([np.ones_like(y), logit_ref, logit_cand])
    
    try:
        model = sm.Logit(y, X).fit(disp=False, cov_type="HAC", cov_kwds={"maxlags": 5})
        params = model.params
        pvalues = model.pvalues
        bse = model.bse
        
        # Test 1: Does Reference encompass Candidate? (H0: beta_cand == 0)
        p_cand_adds_info = pvalues[2]
        # Test 2: Does Candidate encompass Reference? (H0: beta_ref == 0)
        p_ref_adds_info = pvalues[1]
        
        return {
            "candidate": candidate_name,
            "reference": reference_name,
            "beta_ref": float(params[1]),
            "se_ref": float(bse[1]),
            "p_ref": float(p_ref_adds_info),
            "beta_cand": float(params[2]),
            "se_cand": float(bse[2]),
            "p_cand": float(p_cand_adds_info),
            "reference_encompasses": bool(p_cand_adds_info > 0.05),
            "candidate_encompasses": bool(p_ref_adds_info > 0.05)
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 80)
    print("RUNNING ADVANCED STATISTICAL COMPETITIVENESS & EQUALITY TESTS")
    print("=" * 80)
    
    sports, market = load_canonical_data()
    print(f"Loaded datasets: Sports N={len(sports)}, Market N={len(market)}")
    
    lr_col = "p_lr_ts_standard_seed42"
    mlp_col = "p_mlp_ts_standard_seed42"
    gl_col = "p_player_gl"
    
    # =========================================================================
    # 1. NON-INFERIORITY TESTS
    # =========================================================================
    print("\n" + "#" * 80)
    print("1. NON-INFERIORITY TESTS (Test niegorszości)")
    print("H0: Delta LogLoss >= +epsilon (Model ustępuje rynkowi o co najmniej epsilon)")
    print("H1: Delta LogLoss < +epsilon (Model jest konkurencyjny / niegorszy)")
    print("#" * 80)
    
    ni_pairs = [
        (market, lr_col, "market_close", "LR ElasticNet", "Market CLOSE", "Rynek (N=3229)"),
        (market, mlp_col, "market_close", "MLP", "Market CLOSE", "Rynek (N=3229)"),
        (market, lr_col, "market_open", "LR ElasticNet", "Market OPEN", "Rynek (N=3229)"),
        (market, mlp_col, "market_open", "MLP", "Market OPEN", "Rynek (N=3229)"),
        (sports, lr_col, mlp_col, "LR ElasticNet", "MLP", "Sport (N=11550)"),
        (market, lr_col, mlp_col, "LR ElasticNet", "MLP", "Rynek (N=3229)"),
        (sports, lr_col, gl_col, "LR ElasticNet", "Player Glicko-2", "Sport (N=11550)"),
        (sports, mlp_col, gl_col, "MLP", "Player Glicko-2", "Sport (N=11550)"),
    ]
    
    margins = [0.001, 0.002, 0.003, 0.004, 0.005, 0.010]
    ni_results = []
    
    for df_sub, c_col, r_col, c_name, r_name, s_name in ni_pairs:
        res = run_non_inferiority_tests(df_sub, c_col, r_col, c_name, r_name, s_name, margins=margins)
        ni_results.append(res)
        print(f"\n>>> {c_name} vs {r_name} [{s_name}]")
        print(f"    Observed Delta LL: {res['observed_delta']:+.6f} (90% CI: [{res['ci_90'][0]:.4f}; {res['ci_90'][1]:.4f}])")
        for eps in margins:
            m_res = res["margins"][eps]
            verdict = "NIEGORSZY (Konkurencyjny) ✓" if m_res["non_inferior"] else "Brak konkluzji ✗"
            print(f"    eps = {eps:.3f} ({int(eps*1000)} promili): p_value = {m_res['p_value']:.4f}  --> {verdict}")

    # =========================================================================
    # 2. DIEBOLD-MARIANO & HARVEY-LEYBOURNE-NEWBOLD TESTS
    # =========================================================================
    print("\n" + "#" * 80)
    print("2. DIEBOLD-MARIANO (DM) & HARVEY-LEYBOURNE-NEWBOLD (HLN) TESTS")
    print("H0: E[Loss_candidate - Loss_reference] = 0 (Równorzędna zdolność predykcyjna)")
    print("Newey-West HAC standard errors (lag h=1)")
    print("#" * 80)
    
    dm_pairs = [
        (market, lr_col, "market_close", "LR ElasticNet", "Market CLOSE", "Rynek (N=3229)"),
        (market, mlp_col, "market_close", "MLP", "Market CLOSE", "Rynek (N=3229)"),
        (market, lr_col, "market_open", "LR ElasticNet", "Market OPEN", "Rynek (N=3229)"),
        (market, mlp_col, "market_open", "MLP", "Market OPEN", "Rynek (N=3229)"),
        (sports, lr_col, mlp_col, "LR ElasticNet", "MLP", "Sport (N=11550)"),
        (market, lr_col, mlp_col, "LR ElasticNet", "MLP", "Rynek (N=3229)"),
        (sports, lr_col, gl_col, "LR ElasticNet", "Player Glicko-2", "Sport (N=11550)"),
        (sports, mlp_col, gl_col, "MLP", "Player Glicko-2", "Sport (N=11550)"),
    ]
    
    for df_sub, c_col, r_col, c_name, r_name, s_name in dm_pairs:
        y = df_sub["y_true"].to_numpy()
        l_cand = binary_log_loss_vector(y, df_sub[c_col].to_numpy())
        l_ref = binary_log_loss_vector(y, df_sub[r_col].to_numpy())
        dm = diebold_mariano_test(l_cand, l_ref)
        
        print(f"\n>>> {c_name} vs {r_name} [{s_name}]:")
        print(f"    Mean diff (Delta LL): {dm['mean_diff']:+.6f} (HAC SE: {dm['hac_se']:.6f})")
        print(f"    DM Stat: {dm['dm_stat']:+.4f} | p-value (dwustronny): {dm['p_two_sided']:.4f}")
        print(f"    HLN Stat: {dm['hln_stat']:+.4f} | p-value (dwustronny): {dm['p_hln_two_sided']:.4f}")
        print(f"    p-value (jednostronny, H1: kandydat lepszy): {dm['p_candidate_better']:.4f}")
        
        if dm['p_two_sided'] >= 0.05:
            verdict = "RÓWNA ZDOLNOŚĆ PREDYKCYJNA (Equal Predictive Ability - brak istotnej różnicy)"
        elif dm['mean_diff'] < 0:
            verdict = "KANDYDAT STATYSTYCZNIE ISTOTNIE LEPSZY"
        else:
            verdict = "ODNIESIENIE STATYSTYCZNIE ISTOTNIE LEPSZE"
        print(f"    Konkluzja przy alpha=0.05: {verdict}")

    # =========================================================================
    # 3. MODEL CONFIDENCE SET (MCS)
    # =========================================================================
    print("\n" + "#" * 80)
    print("3. MODEL CONFIDENCE SET (MCS, Hansen et al. 2011)")
    print("Wyznaczanie zbioru modeli dominujących (Superior Set of Models)")
    print("#" * 80)
    
    # A. Market Common Sample (N=3229)
    y_m = market["y_true"].to_numpy()
    m_models = ["LR ElasticNet", "MLP", "Market CLOSE", "Market OPEN", "Player Glicko-2"]
    m_cols = [lr_col, mlp_col, "market_close", "market_open", gl_col]
    loss_matrix_market = np.column_stack([binary_log_loss_vector(y_m, market[c].to_numpy()) for c in m_cols])
    
    print("\n--- MCS na wspólnej próbie rynkowej (N=3229) ---")
    mcs_market = model_confidence_set(loss_matrix_market, m_models, alpha=0.05)
    for r in mcs_market:
        status_95 = "W ZBIORZE MCS 95% ✓ (Lider)" if r["in_mcs_95"] else "ODRZUCONY ✗ (Gorszy)"
        status_90 = "W ZBIORZE MCS 90% ✓ (Lider)" if r["in_mcs_90"] else "ODRZUCONY ✗ (Gorszy)"
        print(f"  {r['model']:18s} | LogLoss: {r['mean_loss']:.6f} | MCS p-val: {r['mcs_p_value']:.4f} | {status_95}")
        
    # B. Sports Full Sample (N=11550)
    y_s = sports["y_true"].to_numpy()
    s_models = ["LR ElasticNet", "MLP", "Player Glicko-2"]
    s_cols = [lr_col, mlp_col, gl_col]
    loss_matrix_sports = np.column_stack([binary_log_loss_vector(y_s, sports[c].to_numpy()) for c in s_cols])
    
    print("\n--- MCS na pełnej próbie sportowej (N=11550) ---")
    mcs_sports = model_confidence_set(loss_matrix_sports, s_models, alpha=0.05)
    for r in mcs_sports:
        status_95 = "W ZBIORZE MCS 95% ✓ (Lider)" if r["in_mcs_95"] else "ODRZUCONY ✗ (Gorszy)"
        print(f"  {r['model']:18s} | LogLoss: {r['mean_loss']:.6f} | MCS p-val: {r['mcs_p_value']:.4f} | {status_95}")

    # =========================================================================
    # 4. FORECAST ENCOMPASSING TESTS
    # =========================================================================
    print("\n" + "#" * 80)
    print("4. FORECAST ENCOMPASSING TESTS (Harvey, Leybourne, Newbold 1998)")
    print("Czy jedno źródło prognoz pochłania informacyjnie drugie?")
    print("#" * 80)
    
    enc_pairs = [
        (market, lr_col, "market_close", "LR ElasticNet", "Market CLOSE"),
        (market, mlp_col, "market_close", "MLP", "Market CLOSE"),
        (market, lr_col, "market_open", "LR ElasticNet", "Market OPEN"),
        (market, mlp_col, "market_open", "MLP", "Market OPEN"),
        (sports, lr_col, gl_col, "LR ElasticNet", "Player Glicko-2"),
        (sports, mlp_col, gl_col, "MLP", "Player Glicko-2"),
    ]
    
    for df_sub, c_col, r_col, c_name, r_name in enc_pairs:
        y_sub = df_sub["y_true"].to_numpy()
        enc = forecast_encompassing_test(y_sub, df_sub[c_col].to_numpy(), df_sub[r_col].to_numpy(), c_name, r_name)
        print(f"\n>>> {c_name} vs {r_name}:")
        print(f"    Współczynnik {r_name}: beta = {enc['beta_ref']:+.4f} (SE: {enc['se_ref']:.4f}, p = {enc['p_ref']:.4e})")
        print(f"    Współczynnik {c_name}: beta = {enc['beta_cand']:+.4f} (SE: {enc['se_cand']:.4f}, p = {enc['p_cand']:.4e})")
        
        cand_adds = enc['p_cand'] < 0.05
        ref_adds = enc['p_ref'] < 0.05
        
        if cand_adds and ref_adds:
            interp = f"KOMPLEMENTARNOŚĆ INFORMACYJNA (Oba źródła wnoszą unikalną informację; {r_name} NIE pochłania {c_name}!)"
        elif cand_adds and not ref_adds:
            interp = f"{c_name} POCHŁANIA {r_name} (Całkowita dominacja informacyjna modelu!)"
        elif not cand_adds and ref_adds:
            interp = f"{r_name} POCHŁANIA {c_name} ({r_name} zawiera całą informację modelu)"
        else:
            interp = "Brak jednoznacznego rozstrzygnięcia"
        print(f"    Wniosek: {interp}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
