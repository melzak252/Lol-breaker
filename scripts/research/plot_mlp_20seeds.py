"""Generate publication-quality figure for 20-seed MLP ensemble convergence and ablation."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/08_reporting/mlp_20seeds/cumulative_ensemble.csv"
OUT_PDF = ROOT / "thesis/figures/mlp_20seeds_ensemble.pdf"
OUT_PNG = ROOT / "thesis/figures/mlp_20seeds_ensemble.png"

def main():
    df = pd.read_csv(CSV_PATH)
    print("Loaded data:")
    print(df)

    plt.rcParams.update({
        'font.size': 10,
        'font.sans-serif': 'DejaVu Sans',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'figure.titlesize': 13,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Panel (a): LogLoss vs Ensemble Size M
    M = df["ensemble_size_M"]
    ll_full = df["ll_full"]
    ll_no_ctx = df["ll_no_context"]

    ax1.plot(M, ll_full, 'o-', color='#1f77b4', label='MLP pełny (W20)', zorder=4)
    ax1.plot(M, ll_no_ctx, 's--', color='#7f7f7f', label='MLP bez kontekstu (bez W20)', zorder=3)
    ax1.axhline(0.558374, color='#d62728', linestyle=':', linewidth=1.5, label='LR W20 (baseline = 0,5584)', zorder=2)

    ax1.set_xlabel('Liczba ziaren w zespole ($M$)')
    ax1.set_ylabel('LogLoss na pełnej próbie (mniej = lepiej)')
    ax1.set_title('(a) Błąd LogLoss a wielkość zespołu $M$')
    ax1.set_xticks([1, 2, 3, 5, 10, 15, 20])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='center right', frameon=True, facecolor='white', framealpha=0.9)

    # Panel (b): Delta LogLoss and 95% Confidence Band
    delta = df["delta"] * 1000 # in 10^-3 scale or raw
    ci_lo = df["ci_lower"] * 1000
    ci_hi = df["ci_upper"] * 1000

    ax2.plot(M, delta, 'D-', color='#2ca02c', label=r'$\Delta$ LogLoss (Bez W20 minus Pełny)', zorder=4)
    ax2.fill_between(M, ci_lo, ci_hi, color='#2ca02c', alpha=0.2, label='95% przedział ufności (bootstrap)', zorder=2)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.0, zorder=1)
    ax2.axhline(0.003204 * 1000, color='#d62728', linestyle=':', linewidth=1.5, label=r'$\Delta$ LogLoss w modelu LR (+0,0032)', zorder=3)

    ax2.set_xlabel('Liczba ziaren w zespole ($M$)')
    ax2.set_ylabel(r'Wzrost straty $\Delta$ LogLoss [$\times 10^{-3}$] (więcej = większy zysk z W20)')
    ax2.set_title('(b) Zysk z cech W20 i przedział ufności')
    ax2.set_xticks([1, 2, 3, 5, 10, 15, 20])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, bbox_inches='tight')
    print(f"Generated {OUT_PDF} and {OUT_PNG}")

if __name__ == '__main__':
    main()
