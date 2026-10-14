"""EXP-066: bootstrap/test the current best thesis hybrid vs bookmaker market.

Uses the EXP-060/061 common sample with EXP-039 probabilities aligned to
canonical team A and bookmaker no-vig open/mid/close probabilities. The tested
production-style model is:

    p_hybrid = alpha * temperature(EXP039, T) + (1 - alpha) * market

Default alpha/T match the current production thesis hybrid a0.35-t0.80. For each
market timing (open/mid/close), the script computes paired per-match LogLoss
differences:

    ΔLogLoss = market_loss - hybrid_loss

Positive Δ means the hybrid has lower LogLoss than the market on the same match.
It reports paired t-test, sign test, permutation sign-flip test, ordinary match
bootstrap, and monthly block bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from betting_app.services.thesis_inference_service import apply_temperature_probability

EPS = 1e-15


@dataclass
class ComparisonResult:
    market: str
    n: int
    n_months: int
    market_logloss: float
    hybrid_logloss: float
    exp039_logloss: float
    delta_market_minus_hybrid: float
    delta_market_minus_exp039: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_p_one_sided: float
    block_bootstrap_ci_low: float
    block_bootstrap_ci_high: float
    block_bootstrap_p_one_sided: float
    paired_t_stat: float | None
    paired_t_p_one_sided: float | None
    sign_test_wins: int
    sign_test_losses: int
    sign_test_p_one_sided: float
    permutation_p_one_sided: float
    significant_05_block_bootstrap: bool
    significant_05_permutation: bool


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1 - EPS)
    y = y.astype(float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _mean_logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(_log_loss(y, p).mean())


def _student_t_cdf_approx(t: float, df: int) -> float:
    """Normal approximation fallback; enough for report context if scipy unavailable."""
    # For df in this use-case (hundreds), normal approximation is very close.
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def _paired_t_p_one_sided(diff: np.ndarray) -> tuple[float | None, float | None]:
    n = len(diff)
    if n < 2:
        return None, None
    sd = float(diff.std(ddof=1))
    if sd <= 0:
        return (math.inf if diff.mean() > 0 else -math.inf, 0.0 if diff.mean() > 0 else 1.0)
    t = float(diff.mean() / (sd / math.sqrt(n)))
    try:
        from scipy import stats  # type: ignore
        p = float(1.0 - stats.t.cdf(t, df=n - 1))
    except Exception:
        p = float(1.0 - _student_t_cdf_approx(t, n - 1))
    return t, p


def _binom_tail_p_ge(k: int, n: int, p: float = 0.5) -> float:
    # Exact upper tail for n≈400 is safe using lgamma accumulation.
    if n <= 0:
        return 1.0
    logs = []
    for i in range(k, n + 1):
        logs.append(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log(1 - p))
    m = max(logs)
    return float(min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs)))


def _bootstrap_mean(diff: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    return diff[idx].mean(axis=1)


def _permutation_sign_flip(diff: np.ndarray, rng: np.random.Generator, n_perm: int) -> np.ndarray:
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diff)))
    return (signs * diff).mean(axis=1)


def _block_bootstrap(df: pd.DataFrame, diff_col: str, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    month_summary = (
        df.groupby('month', as_index=False)
        .agg(n=('match_id', 'size'), diff_sum=(diff_col, 'sum'))
        .sort_values('month')
    )
    n_blocks = len(month_summary)
    if n_blocks == 0:
        return np.array([], dtype=float)
    idx = np.arange(n_blocks)
    counts = month_summary['n'].to_numpy(float)
    sums = month_summary['diff_sum'].to_numpy(float)
    sampled = rng.choice(idx, size=(n_boot, n_blocks), replace=True)
    return sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)


def _prepare(df: pd.DataFrame, market: str, alpha: float, temperature: float) -> pd.DataFrame:
    market_col = f'market_{market}_p_a_novig'
    need = ['canonical_match_id', 'start_time_normalized', 'y_team_a', 'exp039_prob_team_a', market_col]
    sub = df[need].dropna().copy()
    sub = sub[(sub[market_col] > 0) & (sub[market_col] < 1) & (sub['exp039_prob_team_a'] > 0) & (sub['exp039_prob_team_a'] < 1)]
    sub['match_id'] = sub['canonical_match_id'].astype(int)
    sub['month'] = pd.to_datetime(sub['start_time_normalized'], utc=True, format='mixed').dt.strftime('%Y-%m')
    y = sub['y_team_a'].to_numpy(int)
    market_p = sub[market_col].to_numpy(float)
    exp039 = sub['exp039_prob_team_a'].to_numpy(float)
    exp039_t = np.array([apply_temperature_probability(float(p), temperature) for p in exp039], dtype=float)
    hybrid = alpha * exp039_t + (1.0 - alpha) * market_p
    sub['y'] = y
    sub['market_prob'] = market_p
    sub['exp039_prob'] = exp039
    sub['hybrid_prob'] = hybrid
    sub['market_loss'] = _log_loss(y, market_p)
    sub['hybrid_loss'] = _log_loss(y, hybrid)
    sub['exp039_loss'] = _log_loss(y, exp039)
    sub['diff_market_minus_hybrid'] = sub['market_loss'] - sub['hybrid_loss']
    sub['diff_market_minus_exp039'] = sub['market_loss'] - sub['exp039_loss']
    return sub


def compare(sub: pd.DataFrame, market: str, rng: np.random.Generator, n_boot: int, n_perm: int) -> ComparisonResult:
    diff = sub['diff_market_minus_hybrid'].to_numpy(float)
    obs = float(diff.mean())
    boot = _bootstrap_mean(diff, rng, n_boot)
    block = _block_bootstrap(sub, 'diff_market_minus_hybrid', rng, n_boot)
    perm = _permutation_sign_flip(diff, rng, n_perm)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    bci_low, bci_high = np.percentile(block, [2.5, 97.5]) if len(block) else (math.nan, math.nan)
    t, tp = _paired_t_p_one_sided(diff)
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    non_ties = wins + losses
    sign_p = _binom_tail_p_ge(wins, non_ties) if non_ties else 1.0
    return ComparisonResult(
        market=market,
        n=int(len(sub)),
        n_months=int(sub['month'].nunique()),
        market_logloss=_mean_logloss(sub['y'].to_numpy(int), sub['market_prob'].to_numpy(float)),
        hybrid_logloss=_mean_logloss(sub['y'].to_numpy(int), sub['hybrid_prob'].to_numpy(float)),
        exp039_logloss=_mean_logloss(sub['y'].to_numpy(int), sub['exp039_prob'].to_numpy(float)),
        delta_market_minus_hybrid=obs,
        delta_market_minus_exp039=float(sub['diff_market_minus_exp039'].mean()),
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        bootstrap_p_one_sided=float((np.sum(boot <= 0.0) + 1) / (len(boot) + 1)),
        block_bootstrap_ci_low=float(bci_low),
        block_bootstrap_ci_high=float(bci_high),
        block_bootstrap_p_one_sided=float((np.sum(block <= 0.0) + 1) / (len(block) + 1)) if len(block) else 1.0,
        paired_t_stat=None if t is None or not math.isfinite(t) else float(t),
        paired_t_p_one_sided=None if tp is None else float(tp),
        sign_test_wins=wins,
        sign_test_losses=losses,
        sign_test_p_one_sided=sign_p,
        permutation_p_one_sided=float((np.sum(perm >= obs) + 1) / (len(perm) + 1)),
        significant_05_block_bootstrap=bool(len(block) and bci_low > 0.0),
        significant_05_permutation=bool(((np.sum(perm >= obs) + 1) / (len(perm) + 1)) < 0.05 and obs > 0),
    )


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='reports/exp039_db_market_backtest_v2/exp039_market_common.csv')
    ap.add_argument('--alpha', type=float, default=0.35)
    ap.add_argument('--temperature', type=float, default=0.80)
    ap.add_argument('--n-bootstrap', type=int, default=10000)
    ap.add_argument('--n-permutation', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--json-output', default='reports/exp066_best_model_vs_market_bootstrap/summary.json')
    args = ap.parse_args(list(argv) if argv is not None else None)

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.input)
    results = []
    samples = {}
    for market in ['open', 'mid', 'close']:
        sub = _prepare(df, market, args.alpha, args.temperature)
        res = compare(sub, market, rng, args.n_bootstrap, args.n_permutation)
        results.append(asdict(res))
        samples[market] = sub[[
            'match_id', 'start_time_normalized', 'month', 'y', 'market_prob', 'exp039_prob', 'hybrid_prob',
            'market_loss', 'hybrid_loss', 'exp039_loss', 'diff_market_minus_hybrid', 'diff_market_minus_exp039'
        ]]

    output = {
        'experiment_id': 'EXP-066',
        'model': 'Hybrid-Thesis-Market/a0.35-t0.80',
        'model_formula': 'p_hybrid = alpha * temperature(EXP039, T) + (1-alpha) * market_probability',
        'alpha': args.alpha,
        'temperature': args.temperature,
        'input': args.input,
        'n_bootstrap': args.n_bootstrap,
        'n_permutation': args.n_permutation,
        'seed': args.seed,
        'delta_definition': 'market_logloss - hybrid_logloss; positive means hybrid is better/lower LogLoss',
        'results': results,
    }
    out = Path(args.json_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    for market, sub in samples.items():
        sub.to_csv(out.parent / f'{market}_per_match_losses.csv', index=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
