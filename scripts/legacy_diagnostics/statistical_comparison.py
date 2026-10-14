"""
Statistical comparison of Hybrid-Thesis-Market vs Market-only predictions.
Computes per-match LogLoss for both models and runs paired statistical tests.
"""
import psycopg2
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, '/app')
from betting_app.services.canonical_match_service import align_snapshot_odds

DB_CONFIG = {
    'host': '192.168.1.17',
    'port': 5432,
    'user': 'betting',
    'password': 'betting_local_password',
    'dbname': 'betting',
}

ALPHA = 0.50  # hybrid alpha
TEMPERATURE = 0.80  # temperature scaling

def apply_temperature(prob, temperature):
    """Apply temperature scaling to probability."""
    if temperature == 1.0 or prob is None:
        return prob
    logit = np.log(prob / (1 - prob))
    scaled_logit = logit / temperature
    return 1.0 / (1.0 + np.exp(-scaled_logit))

def logloss(y_true, y_prob):
    """Compute log loss for a single observation."""
    eps = 1e-15
    p = np.clip(y_prob, eps, 1 - eps)
    if y_true == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get finished matches with results
    cur.execute("""
        SELECT cm.id, cm.winner_side, cm.start_time_normalized
        FROM canonical_matches cm
        WHERE cm.status IN ('finished', 'completed')
          AND cm.winner_side IS NOT NULL
          AND cm.start_time_normalized IS NOT NULL
          AND cm.start_time_normalized > '2026-04-01'
        ORDER BY cm.id
    """)
    matches = cur.fetchall()
    print(f"Total finished matches: {len(matches)}")

    # Get latest thesis predictions per match
    cur.execute("""
        WITH ranked AS (
            SELECT cp.canonical_match_id, cp.prob_a, cp.prob_b,
                   ROW_NUMBER() OVER (PARTITION BY cp.canonical_match_id 
                                      ORDER BY cp.predicted_at DESC NULLS LAST, cp.id DESC) as rn
            FROM canonical_predictions cp
            WHERE cp.model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
              AND cp.model_version = 'exp-039'
        )
        SELECT canonical_match_id, prob_a, prob_b
        FROM ranked WHERE rn = 1
    """)
    thesis_preds = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    print(f"Thesis predictions available: {len(thesis_preds)}")

    # For each match, compute per-match LogLoss for market and hybrid
    # We'll use ALL pre-match odds snapshots (not binned) and average per bookmaker per match
    results = []

    for match_id, winner_side, start_time in matches:
        if match_id not in thesis_preds:
            continue
        
        thesis_prob_a, thesis_prob_b = thesis_preds[match_id]
        if thesis_prob_a is None or thesis_prob_b is None:
            continue

        # Get pre-match odds snapshots (not live)
        cur.execute("""
            SELECT os.bookmaker_id, os.odds_a, os.odds_b, os.scraped_at,
                   os.raw_team_a, os.raw_team_b,
                   cm.team_a_name, cm.team_b_name, cm.league, cm.start_time_normalized
            FROM odds_snapshots os
            JOIN canonical_matches cm ON cm.id = os.canonical_match_id
            WHERE os.canonical_match_id = %s
              AND os.market_type = 'match_winner'
              AND os.is_live = 0
              AND os.odds_a > 1 AND os.odds_b > 1
            ORDER BY os.scraped_at DESC
        """, (match_id,))
        snapshots = cur.fetchall()

        if not snapshots:
            continue

        # Average odds per bookmaker (latest snapshot per bookmaker)
        bookmaker_probs = {}
        seen_books = set()
        for bookmaker_id, odds_a, odds_b, scraped_at, raw_a, raw_b, can_a, can_b, league, start_at in snapshots:
            if bookmaker_id not in seen_books:
                seen_books.add(bookmaker_id)
                aligned = align_snapshot_odds(
                    can_a, can_b, raw_a, raw_b, odds_a, odds_b,
                    league=league, source_system="bookmaker", match_date=start_at,
                )
                if aligned is None:
                    continue
                odds_a, odds_b = aligned
                # Convert odds to implied probabilities
                implied_a = 1.0 / odds_a
                implied_b = 1.0 / odds_b
                # Normalize (remove overround)
                total = implied_a + implied_b
                bookmaker_probs[bookmaker_id] = (implied_a / total, implied_b / total)

        if not bookmaker_probs:
            continue

        # Average market probabilities across bookmakers
        market_prob_a = np.mean([p[0] for p in bookmaker_probs.values()])
        market_prob_b = np.mean([p[1] for p in bookmaker_probs.values()])

        # Apply temperature to market probs
        temp_market_prob_a = apply_temperature(market_prob_a, TEMPERATURE)
        temp_market_prob_b = apply_temperature(market_prob_b, TEMPERATURE)

        # Hybrid probabilities
        hybrid_prob_a = ALPHA * thesis_prob_a + (1 - ALPHA) * temp_market_prob_a
        hybrid_prob_b = ALPHA * thesis_prob_b + (1 - ALPHA) * temp_market_prob_b

        # Normalize hybrid
        hybrid_total = hybrid_prob_a + hybrid_prob_b
        hybrid_prob_a /= hybrid_total
        hybrid_prob_b /= hybrid_total

        # Determine y_true (1 if team A won, 0 if team B won)
        y_true = 1 if winner_side == 'a' else 0

        # LogLoss for market
        market_prob_winner = market_prob_a if winner_side == 'a' else market_prob_b
        ll_market = logloss(1, market_prob_winner)

        # LogLoss for hybrid
        hybrid_prob_winner = hybrid_prob_a if winner_side == 'a' else hybrid_prob_b
        ll_hybrid = logloss(1, hybrid_prob_winner)

        # LogLoss for thesis pure
        thesis_prob_winner = thesis_prob_a if winner_side == 'a' else thesis_prob_b
        ll_thesis = logloss(1, thesis_prob_winner)

        results.append({
            'match_id': match_id,
            'winner_side': winner_side,
            'n_bookmakers': len(bookmaker_probs),
            'll_market': ll_market,
            'll_hybrid': ll_hybrid,
            'll_thesis': ll_thesis,
            'market_prob_winner': market_prob_winner,
            'hybrid_prob_winner': hybrid_prob_winner,
            'thesis_prob_winner': thesis_prob_winner,
        })

    conn.close()

    if not results:
        print("No results to analyze!")
        return

    n = len(results)
    ll_market = np.array([r['ll_market'] for r in results])
    ll_hybrid = np.array([r['ll_hybrid'] for r in results])
    ll_thesis = np.array([r['ll_thesis'] for r in results])

    print(f"\n{'='*60}")
    print(f"Per-match LogLoss comparison (n={n} matches)")
    print(f"{'='*60}")
    print(f"Market only:  mean={ll_market.mean():.4f}, std={ll_market.std():.4f}")
    print(f"Hybrid:       mean={ll_hybrid.mean():.4f}, std={ll_hybrid.std():.4f}")
    print(f"Thesis pure:  mean={ll_thesis.mean():.4f}, std={ll_thesis.std():.4f}")

    # Paired t-test: Market vs Hybrid
    diff_mh = ll_market - ll_hybrid  # positive = hybrid better
    t_stat_mh, p_val_mh = stats.ttest_rel(ll_market, ll_hybrid)
    print(f"\n{'='*60}")
    print(f"Paired t-test: Market vs Hybrid")
    print(f"{'='*60}")
    print(f"Mean diff (market - hybrid): {diff_mh.mean():.4f}")
    print(f"Std diff: {diff_mh.std():.4f}")
    print(f"t-statistic: {t_stat_mh:.4f}")
    print(f"p-value (two-sided): {p_val_mh:.6f}")
    print(f"p-value (one-sided, hybrid better): {p_val_mh/2:.6f}")
    print(f"Effect size (Cohen's d): {diff_mh.mean() / diff_mh.std():.4f}")

    # Paired t-test: Market vs Thesis
    diff_mt = ll_market - ll_thesis
    t_stat_mt, p_val_mt = stats.ttest_rel(ll_market, ll_thesis)
    print(f"\n{'='*60}")
    print(f"Paired t-test: Market vs Thesis")
    print(f"{'='*60}")
    print(f"Mean diff (market - thesis): {diff_mt.mean():.4f}")
    print(f"Std diff: {diff_mt.std():.4f}")
    print(f"t-statistic: {t_stat_mt:.4f}")
    print(f"p-value (two-sided): {p_val_mt:.6f}")
    print(f"p-value (one-sided, thesis better): {p_val_mt/2:.6f}")
    print(f"Effect size (Cohen's d): {diff_mt.mean() / diff_mt.std():.4f}")

    # Paired t-test: Thesis vs Hybrid
    diff_th = ll_thesis - ll_hybrid
    t_stat_th, p_val_th = stats.ttest_rel(ll_thesis, ll_hybrid)
    print(f"\n{'='*60}")
    print(f"Paired t-test: Thesis vs Hybrid")
    print(f"{'='*60}")
    print(f"Mean diff (thesis - hybrid): {diff_th.mean():.4f}")
    print(f"Std diff: {diff_th.std():.4f}")
    print(f"t-statistic: {t_stat_th:.4f}")
    print(f"p-value (two-sided): {p_val_th:.6f}")
    print(f"p-value (one-sided, hybrid better): {p_val_th/2:.6f}")
    print(f"Effect size (Cohen's d): {diff_th.mean() / diff_th.std():.4f}")

    # Wilcoxon signed-rank test (non-parametric alternative)
    print(f"\n{'='*60}")
    print(f"Wilcoxon signed-rank tests (non-parametric)")
    print(f"{'='*60}")
    
    w_stat_mh, w_p_mh = stats.wilcoxon(ll_market, ll_hybrid, alternative='greater')
    print(f"Market > Hybrid: W={w_stat_mh:.1f}, p={w_p_mh:.6f}")
    
    w_stat_mt, w_p_mt = stats.wilcoxon(ll_market, ll_thesis, alternative='greater')
    print(f"Market > Thesis: W={w_stat_mt:.1f}, p={w_p_mt:.6f}")
    
    w_stat_th, w_p_th = stats.wilcoxon(ll_thesis, ll_hybrid, alternative='greater')
    print(f"Thesis > Hybrid: W={w_stat_th:.1f}, p={w_p_th:.6f}")

    # Power analysis: what n would we need for significance?
    print(f"\n{'='*60}")
    print(f"Power analysis")
    print(f"{'='*60}")
    d_mh = diff_mh.mean() / diff_mh.std() if diff_mh.std() > 0 else 0
    d_mt = diff_mt.mean() / diff_mt.std() if diff_mt.std() > 0 else 0
    d_th = diff_th.mean() / diff_th.std() if diff_th.std() > 0 else 0
    
    for label, d in [("Market vs Hybrid", d_mh), ("Market vs Thesis", d_mt), ("Thesis vs Hybrid", d_th)]:
        if d > 0:
            # Approximate n needed for 80% power at alpha=0.05 (two-sided)
            # n = (2 * (z_alpha/2 + z_beta)^2) / d^2 for paired test
            from scipy.stats import norm
            z_a = norm.ppf(0.975)  # alpha=0.05 two-sided
            z_b = norm.ppf(0.80)   # power=0.80
            n_needed = ((z_a + z_b)**2) / d**2
            print(f"{label}: Cohen's d={d:.4f}, n needed for 80% power ≈ {n_needed:.0f}")
        else:
            print(f"{label}: Cohen's d={d:.4f}, effect in wrong direction")

    # Per-bin analysis
    print(f"\n{'='*60}")
    print(f"Per-bin analysis (time before match)")
    print(f"{'='*60}")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    bins = [
        ("0-2h", 0, 2),
        ("2-6h", 2, 6),
        ("6-12h", 6, 12),
        ("12-24h", 12, 24),
        ("24-48h", 24, 48),
        ("48h+", 48, 999),
    ]
    
    for label, h_start, h_end in bins:
        bin_results = []
        for r in results:
            match_id = r['match_id']
            cur.execute("""
                SELECT cm.start_time_normalized
                FROM canonical_matches cm WHERE cm.id = %s
            """, (match_id,))
            row = cur.fetchone()
            if not row or not row[0]:
                continue
            
            start_time_str = row[0]
            
            # Get odds snapshots in this time bin
            cur.execute("""
                SELECT os.bookmaker_id, os.odds_a, os.odds_b,
                       os.raw_team_a, os.raw_team_b,
                       cm.team_a_name, cm.team_b_name, cm.league, cm.start_time_normalized
                FROM odds_snapshots os
                JOIN canonical_matches cm ON cm.id = os.canonical_match_id
                WHERE os.canonical_match_id = %s
                  AND os.market_type = 'match_winner'
                  AND os.is_live = 0
                  AND os.odds_a > 1 AND os.odds_b > 1
                  AND EXTRACT(EPOCH FROM (%s::timestamp - os.scraped_at)) / 3600 BETWEEN %s AND %s
            """, (match_id, start_time_str, h_start, h_end))
            
            snapshots = cur.fetchall()
            if not snapshots:
                continue
            
            bookmaker_probs = {}
            seen_books = set()
            for bookmaker_id, odds_a, odds_b, raw_a, raw_b, can_a, can_b, league, start_at in snapshots:
                if bookmaker_id not in seen_books:
                    seen_books.add(bookmaker_id)
                    aligned = align_snapshot_odds(
                        can_a, can_b, raw_a, raw_b, odds_a, odds_b,
                        league=league, source_system="bookmaker", match_date=start_at,
                    )
                    if aligned is None:
                        continue
                    odds_a, odds_b = aligned
                    implied_a = 1.0 / odds_a
                    implied_b = 1.0 / odds_b
                    total = implied_a + implied_b
                    bookmaker_probs[bookmaker_id] = (implied_a / total, implied_b / total)
            
            if not bookmaker_probs:
                continue
            
            market_prob_a = np.mean([p[0] for p in bookmaker_probs.values()])
            market_prob_b = np.mean([p[1] for p in bookmaker_probs.values()])
            
            temp_market_prob_a = apply_temperature(market_prob_a, TEMPERATURE)
            temp_market_prob_b = apply_temperature(market_prob_b, TEMPERATURE)
            
            thesis_prob_a, thesis_prob_b = thesis_preds[match_id]
            hybrid_prob_a = ALPHA * thesis_prob_a + (1 - ALPHA) * temp_market_prob_a
            hybrid_prob_b = ALPHA * thesis_prob_b + (1 - ALPHA) * temp_market_prob_b
            hybrid_total = hybrid_prob_a + hybrid_prob_b
            hybrid_prob_a /= hybrid_total
            hybrid_prob_b /= hybrid_total
            
            winner_side = r['winner_side']
            market_prob_winner = market_prob_a if winner_side == 'a' else market_prob_b
            hybrid_prob_winner = hybrid_prob_a if winner_side == 'a' else hybrid_prob_b
            
            bin_results.append({
                'll_market': logloss(1, market_prob_winner),
                'll_hybrid': logloss(1, hybrid_prob_winner),
            })
        
        if bin_results:
            ll_m = np.array([b['ll_market'] for b in bin_results])
            ll_h = np.array([b['ll_hybrid'] for b in bin_results])
            diff = ll_m - ll_h
            
            if len(bin_results) >= 2 and diff.std() > 0:
                t_stat, p_val = stats.ttest_rel(ll_m, ll_h)
                d = diff.mean() / diff.std()
                print(f"  {label}: n={len(bin_results)}, Market LL={ll_m.mean():.4f}, Hybrid LL={ll_h.mean():.4f}, "
                      f"diff={diff.mean():.4f}, t={t_stat:.3f}, p={p_val/2:.4f} (one-sided), d={d:.3f}")
            else:
                print(f"  {label}: n={len(bin_results)}, Market LL={ll_m.mean():.4f}, Hybrid LL={ll_h.mean():.4f}, insufficient data for test")
    
    conn.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
