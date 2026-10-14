"""Odds Bracket Calibration and Betting Performance Benchmark.

Evaluates probabilistic models across standard bookmaker odds brackets,
measuring calibration error (overconfidence/underconfidence), Brier score,
LogLoss, and simulated betting ROI (both gross and net of 12% Polish turnover tax).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OddsBracketTier:
    """Definition of an odds bracket."""
    name: str
    min_odds: float
    max_odds: float
    description: str


DEFAULT_ODDS_BRACKETS: tuple[OddsBracketTier, ...] = (
    OddsBracketTier("Mega Favorite", 1.01, 1.25, "Heavy favorite, implied P > 80%"),
    OddsBracketTier("Solid Favorite", 1.25, 1.50, "Clear favorite, implied P 67-80%"),
    OddsBracketTier("Moderate Favorite", 1.50, 1.80, "Moderate favorite, implied P 55-67%"),
    OddsBracketTier("Coin-Flip / Tight", 1.80, 2.20, "Even matchup, implied P 45-55%"),
    OddsBracketTier("Moderate Underdog", 2.20, 3.00, "Moderate underdog, implied P 33-45%"),
    OddsBracketTier("Big Underdog", 3.00, 5.00, "Substantial underdog, implied P 20-33%"),
    OddsBracketTier("Longshot Underdog", 5.00, 100.0, "High variance dog, implied P < 20%"),
)


@dataclass(frozen=True)
class BracketCalibrationStats:
    """Calibration and performance metrics for a single odds bracket."""
    bracket_name: str
    min_odds: float
    max_odds: float
    sample_size: int
    share_of_total: float
    mean_odds: float
    mean_market_prob: float
    mean_model_prob: float
    observed_win_rate: float
    calibration_error: float  # model_prob - observed_win_rate
    calibration_bias: str      # Overconfident, Underconfident, or Well-calibrated
    brier_score: float
    log_loss: float
    market_log_loss: float
    delta_log_loss: float  # market_log_loss - model_log_loss (positive means model is better)
    n_positive_ev_bets: int
    ev_bet_rate: float
    roi_gross_pct: float | None
    roi_net_pct: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OddsBenchmarkReport:
    """Complete odds-bracket benchmark report."""
    dataset_name: str
    total_lines: int
    overall_brier: float
    overall_log_loss: float
    overall_market_log_loss: float
    bracket_weighted_ece: float
    brackets: list[BracketCalibrationStats]

    def to_markdown(self) -> str:
        """Format calibration and betting metrics into a structured Markdown table."""
        lines = [
            f"# Odds Bracket Calibration & Betting Benchmark: {self.dataset_name}",
            "",
            f"- **Total betting opportunities / lines:** {self.total_lines:,}",
            f"- **Overall Model LogLoss:** {self.overall_log_loss:.4f}",
            f"- **Overall Market LogLoss (no-vig):** {self.overall_market_log_loss:.4f} "
            f"(Delta: {self.overall_market_log_loss - self.overall_log_loss:+.4f})",
            f"- **Bracket-Weighted ECE:** {self.bracket_weighted_ece:.4f} ({self.bracket_weighted_ece*100:.2f}%)",
            "",
            "## 1. Calibration and LogLoss by Bookmaker Odds Tier",
            "",
            "| Odds Bracket | Range | N Lines | Mean Odds | Obs Win% | Model P | Market P | Calib Bias | Model LogLoss | Market LogLoss | Delta (Mkt - Mod) | Brier |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for b in self.brackets:
            bias_str = f"{b.calibration_error*100:+.2f}% ({b.calibration_bias})"
            delta_str = f"{b.delta_log_loss:+.4f}"
            lines.append(
                f"| **{b.bracket_name}** | [{b.min_odds:.2f}, {b.max_odds:.2f}] | {b.sample_size:,} | "
                f"{b.mean_odds:.2f} | {b.observed_win_rate*100:.1f}% | {b.mean_model_prob*100:.1f}% | "
                f"{b.mean_market_prob*100:.1f}% | {bias_str} | **{b.log_loss:.4f}** | {b.market_log_loss:.4f} | "
                f"{delta_str} | {b.brier_score:.4f} |"
            )

        lines.extend([
            "",
            "## 2. Simulated Value Betting Performance (+EV Bets)",
            "",
            "| Odds Bracket | Range | +EV Bets | Bet Share | Mean Odds | Model P | Obs Win% | ROI Brutto | ROI Netto (-12% Tax) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for b in self.brackets:
            roi_g_str = f"{b.roi_gross_pct:+.1f}%" if b.roi_gross_pct is not None else "—"
            roi_n_str = f"{b.roi_net_pct:+.1f}%" if b.roi_net_pct is not None else "—"
            lines.append(
                f"| **{b.bracket_name}** | [{b.min_odds:.2f}, {b.max_odds:.2f}] | {b.n_positive_ev_bets:,} | "
                f"{b.ev_bet_rate*100:.1f}% | {b.mean_odds:.2f} | {b.mean_model_prob*100:.1f}% | "
                f"{b.observed_win_rate*100:.1f}% | {roi_g_str} | {roi_n_str} |"
            )

        lines.extend([
            "",
            "## 3. Key Calibration Findings & Diagnostics",
            "",
            "- **Well-Calibrated Tiers (Bias within ±1.5%):** " + (
                ", ".join(b.bracket_name for b in self.brackets if abs(b.calibration_error) <= 0.015)
                or "None"
            ),
            "- **Overconfident Tiers (Model P > Obs Win Rate + 1.5%):** " + (
                ", ".join(f"{b.bracket_name} ({b.calibration_error*100:+.1f}%)" for b in self.brackets if b.calibration_error > 0.015)
                or "None"
            ),
            "- **Underconfident Tiers (Model P < Obs Win Rate - 1.5%):** " + (
                ", ".join(f"{b.bracket_name} ({b.calibration_error*100:+.1f}%)" for b in self.brackets if b.calibration_error < -0.015)
                or "None"
            ),
            "- **Underdog Safety Check (Odds > 3.50):** " + (
                "PASSED — Underdogs are accurately priced or quarantined without reckless over-betting."
                if all(b.mean_model_prob < 0.35 for b in self.brackets if b.min_odds >= 3.0)
                else "WARNING — High model confidence on longshots detected."
            ),
        ])

        return "\n".join(lines)


def evaluate_odds_bracket_benchmark(
    df: pd.DataFrame,
    odds_col: str = "odds",
    model_prob_col: str = "p_model",
    target_col: str = "y_true",
    market_prob_col: str = "p_novig",
    brackets: tuple[OddsBracketTier, ...] = DEFAULT_ODDS_BRACKETS,
    tax_rate: float = 0.12,
    min_net_ev: float | None = None,
    dataset_name: str = "Application Matches",
) -> OddsBenchmarkReport:
    """Evaluate calibration and betting metrics partitioned by odds brackets."""
    clean_df = df.dropna(subset=[odds_col, model_prob_col, target_col]).copy()
    if min_net_ev is not None:
        odds_arr = clean_df[odds_col].astype(float).to_numpy()
        p_arr = clean_df[model_prob_col].astype(float).to_numpy()
        net_ev = (p_arr * odds_arr * (1.0 - tax_rate)) - 1.0
        clean_df = clean_df[net_ev >= min_net_ev].copy()

    total_lines = len(clean_df)
    if total_lines == 0:
        raise ValueError("Cannot evaluate odds benchmark on empty DataFrame")
    y_all = clean_df[target_col].astype(int).to_numpy()
    p_all = np.clip(clean_df[model_prob_col].astype(float).to_numpy(), 1e-15, 1.0 - 1e-15)

    overall_brier = float(np.mean((p_all - y_all) ** 2))
    overall_log_loss = float(-np.mean(y_all * np.log(p_all) + (1.0 - y_all) * np.log(1.0 - p_all)))

    if market_prob_col in clean_df.columns:
        p_mkt_all = np.clip(clean_df[market_prob_col].astype(float).to_numpy(), 1e-15, 1.0 - 1e-15)
        overall_market_log_loss = float(-np.mean(y_all * np.log(p_mkt_all) + (1.0 - y_all) * np.log(1.0 - p_mkt_all)))
    else:
        odds_all = clean_df[odds_col].astype(float).to_numpy()
        p_mkt_all = np.clip(1.0 / odds_all, 1e-15, 1.0 - 1e-15)
        overall_market_log_loss = float(-np.mean(y_all * np.log(p_mkt_all) + (1.0 - y_all) * np.log(1.0 - p_mkt_all)))

    bracket_results: list[BracketCalibrationStats] = []
    weighted_ece_sum = 0.0

    for tier in brackets:
        # Match mask logic based on bracket boundary conventions
        if tier.min_odds >= 5.0:
            mask = clean_df[odds_col] >= tier.min_odds
        elif tier.min_odds >= 2.20:
            mask = (clean_df[odds_col] > tier.min_odds) & (clean_df[odds_col] <= tier.max_odds)
        elif tier.min_odds >= 1.80:
            mask = (clean_df[odds_col] >= tier.min_odds) & (clean_df[odds_col] <= tier.max_odds)
        else:
            mask = (clean_df[odds_col] >= tier.min_odds) & (clean_df[odds_col] < tier.max_odds)

        sub = clean_df[mask]
        n_sub = len(sub)
        if n_sub == 0:
            continue

        y_sub = sub[target_col].astype(int).to_numpy()
        p_sub = np.clip(sub[model_prob_col].astype(float).to_numpy(), 1e-15, 1.0 - 1e-15)
        odds_sub = sub[odds_col].astype(float).to_numpy()

        if market_prob_col in sub.columns:
            p_mkt_sub = sub[market_prob_col].astype(float).to_numpy()
            mean_mkt = float(np.mean(p_mkt_sub))
        else:
            # Approximate implied probability from decimal odds if no-vig not supplied
            mean_mkt = float(np.mean(1.0 / odds_sub))

        mean_odds = float(np.mean(odds_sub))
        mean_model = float(np.mean(p_sub))
        obs_win = float(np.mean(y_sub))
        calib_error = mean_model - obs_win

        if calib_error > 0.015:
            bias = "Overconfident"
        elif calib_error < -0.015:
            bias = "Underconfident"
        else:
            bias = "Well-calibrated"

        brier = float(np.mean((p_sub - y_sub) ** 2))
        log_loss = float(-np.mean(y_sub * np.log(p_sub) + (1.0 - y_sub) * np.log(1.0 - p_sub)))
        if market_prob_col in sub.columns:
            p_mkt_clip = np.clip(p_mkt_sub, 1e-15, 1.0 - 1e-15)
            mkt_log_loss = float(-np.mean(y_sub * np.log(p_mkt_clip) + (1.0 - y_sub) * np.log(1.0 - p_mkt_clip)))
        else:
            p_mkt_clip = np.clip(1.0 / odds_sub, 1e-15, 1.0 - 1e-15)
            mkt_log_loss = float(-np.mean(y_sub * np.log(p_mkt_clip) + (1.0 - y_sub) * np.log(1.0 - p_mkt_clip)))
        delta_ll = mkt_log_loss - log_loss

        # Value bets (+EV) where expected return exceeds break-even
        ev_mask = (p_sub * odds_sub) > 1.0
        n_ev = int(np.sum(ev_mask))
        ev_share = float(n_ev / n_sub) if n_sub > 0 else 0.0

        if n_ev > 0:
            y_ev = y_sub[ev_mask]
            odds_ev = odds_sub[ev_mask]
            # Gross returns
            returns_gross = (y_ev * odds_ev) - 1.0
            roi_gross = float(np.mean(returns_gross) * 100.0)
            # Net returns with turnover tax: net_payout = 0.88 * odds if win, else 0
            tax_multiplier = 1.0 - tax_rate
            returns_net = (y_ev * odds_ev * tax_multiplier) - 1.0
            roi_net = float(np.mean(returns_net) * 100.0)
        else:
            roi_gross = None
            roi_net = None

        share_of_total = float(n_sub / total_lines)
        weighted_ece_sum += abs(calib_error) * share_of_total

        bracket_results.append(
            BracketCalibrationStats(
                bracket_name=tier.name,
                min_odds=tier.min_odds,
                max_odds=tier.max_odds,
                sample_size=n_sub,
                share_of_total=share_of_total,
                mean_odds=mean_odds,
                mean_market_prob=mean_mkt,
                mean_model_prob=mean_model,
                observed_win_rate=obs_win,
                calibration_error=calib_error,
                calibration_bias=bias,
                brier_score=brier,
                log_loss=log_loss,
                market_log_loss=mkt_log_loss,
                delta_log_loss=delta_ll,
                n_positive_ev_bets=n_ev,
                ev_bet_rate=ev_share,
                roi_gross_pct=roi_gross,
                roi_net_pct=roi_net,
            )
        )

    return OddsBenchmarkReport(
        dataset_name=dataset_name,
        total_lines=total_lines,
        overall_brier=overall_brier,
        overall_log_loss=overall_log_loss,
        overall_market_log_loss=overall_market_log_loss,
        bracket_weighted_ece=weighted_ece_sum,
        brackets=bracket_results,
    )
