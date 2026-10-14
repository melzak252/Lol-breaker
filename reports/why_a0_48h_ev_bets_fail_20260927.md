# Why A0 can predict matches well and still fail on 48-hour EV bets

**Date:** 2026-09-27. **Scope:** interpretation of already saved annual Causal A0 diagnostics, not a new backtest, model fit, or production observation. OPEN and the 48-hour cutoff are separate horizons; this explanation uses the actual `48h` rows. The operational EXP081/market hybrid is not this model.

The [bookmaker/provider research, training audit and experiment recommendation](bookmaker_noise_and_provider_training_20260927.md) remain saved unchanged, with their figures, source ledgers, stronger-reasoning reviews and numerical receipts in `data/08_reporting/bookmaker_noise_research_20260927_200205/`.

## Direct answer

**The observed failure is not that A0 has no predictive information. It is that the probabilities used to select apparent bargains overstate the size of the advantage on those selected outcomes.** Better average match prediction does not guarantee better identification of market errors, and aggregate calibration does not guarantee calibration conditional on price and selection.

There are two separate problems: the selected-bet probabilities are optimistic in the saved sample, and the earlier model-versus-market comparison is not a certified same-information 48-hour test. The latter weakens the claimed operational advantage; it is not a demonstrated cause of the observed losses.

## What happened in the saved 48-hour selection

Source: [selection summary](../data/08_reporting/production_market_calibration_ev5_20260926_210430/ev/selection_summary.csv), filtered to `horizon=48h`, `cohort=common_all_horizons`, `selector=A0`, `tax_rate=0.12`.

The parent population is 275 fixtures shared by all four quote horizons. The rule selects at most one side per fixture with predicted net EV strictly above 5%, using the best stored side-specific quote at the cutoff. It is a raw EV diagnostic, without the application's conservative-probability gates, limits or capital simulation.

| Quantity | Saved result |
|---|---:|
| Selected bets | 82 |
| Mean A0 probability on the selected side | 48.80% |
| Mean market consensus probability on the same side | 33.40% |
| Realized wins | 32/82 = 39.02% |
| Mean predicted net EV | +31.70% |
| Realized flat unit-stake return | −3.39% |
| Observed probability shortfall versus A0 | −9.78 percentage points |
| Return minus predicted EV | −35.09 percentage points |

A0's probabilities imply about 40.02 expected wins, versus 32 observed. The realized win fraction lies between the market and A0 averages. This is compatible with some useful model information and an exaggerated edge; it is not proof that either set of probabilities is correct or that the model has no value.

On exactly these selected outcomes, A0's descriptive ECE10 is 12.21%, versus 4.05% on the complete common population. ECE is noisy and bin-dependent; the more directly relevant evidence is the directional probability and odds-weighted EV gap. See [conditional calibration](../data/08_reporting/production_market_calibration_ev5_20260926_210430/ev/conditional_calibration.csv).

## Why the selection can amplify model errors

An apparent value bet can reflect either a market underestimate or a model overestimate. Selecting the largest positive estimated advantages can enrich the second kind as well as the first. **[INFERENCE]** This selection mechanism is consistent with the observed optimism, but the saved results do not isolate its causal contribution from drift, missing information or other model errors.

The mathematical distinction is:

- Aggregate calibration asks whether `E[Y | p]` is approximately `p`.
- This decision needs calibration conditional on the available price, information horizon and selected subset as well.
- Under the saved stake-tax convention, `EV_hat = p_hat * odds * 0.88 - 1`.
- The realized difference is `return - EV_hat = odds * 0.88 * (Y - p_hat)`.

Consequently, errors on high-odds outcomes have disproportionate financial weight. A model can have a good overall log loss while being too optimistic precisely where the EV rule spends money. Bookmaker miscalibration alone does not identify an exploitable direction or establish that our estimate is closer to the truth at that cutoff.

### The actual 48-hour longshot slice

For odds at least 5.00, this selection contains **22 bets**, A0 mean probability **23.15%**, and **2/22 wins (9.09%)**. The model implies about **5.09 expected wins**. Predicted mean net EV is **+49.36%**, whereas the realized flat return is **−33.24%**.

This is descriptive concentration of the shortfall, not proof that all such bets have negative true EV. The saved seven-day intervals for this small tail's probability and EV gaps include zero. Do not choose an odds cap after inspecting these outcomes and present it as validated. The previously discussed **2/16** slice belongs to **CLOSE**, not 48h. Source: [saved odds slices](../data/08_reporting/production_market_calibration_ev5_20260926_210430/ev/ev_odds_slices.csv).

## Tax can consume a real gross edge, but it does not explain away the EV overstatement

Holding the same 82 selections fixed, their saved net return implies a mechanical before-tax return of **+9.78%**:

`gross_return = (1 + net_return) / 0.88 - 1`.

This is not the separately reselected zero-tax strategy. Under this tax convention, a gross return must exceed about 13.64% to remain positive after tax. Thus positive gross performance can coexist with a net loss.

However, the claimed **+31.70% EV already included the 12% tax**. The gap between that forecast and −3.39% cannot be blamed on forgetting the tax.

For scale, at decimal odds 5.00 the net break-even probability is 22.73%. A claimed +5% net EV requires only 23.86%—a cushion of **1.14 probability percentage points**. Modest probability error can erase a seemingly positive edge. An illustrative estimate of 28% gives +23.2% net EV; if the true probability were 20%, true net EV would instead be −12%. These are arithmetic examples, not additional observed matches.

## The timing comparison is not genuinely 48-hour causal evidence

The saved panel applies the same match-day known-roster A0 forecast to different quote horizons. It does not reconstruct all features and roster knowledge as they existed 48 hours earlier. In the full 343-fixture 48h cohort, only **one** row passes the conservative `strict_prior_day` feature-availability screen; even that does not prove the roster was announced or the forecast issued at that time.

Stored quotes also do not certify executability. These limitations prevent treating historical average score differences as a demonstrated tradable advantage. They must not be used as a post-hoc explanation that supposedly accounts for every losing bet. Source: [original protocol and limitations](../data/08_reporting/production_market_calibration_ev5_20260926_210430/protocol.json) and the existing 2026-09-26 research handoff.

## What is established, and what is not

The saved 82-bet probability gap is negative under both reported dependence assumptions:

| Nominal 95% interval | Monthly blocks | Random-start seven-day blocks |
|---|---:|---:|
| Observed win fraction minus mean A0 probability | [−28.12, −6.87] pp | [−14.64, −0.94] pp |
| Realized return minus predicted EV | [−120.93, −21.32] pp | [−55.66, −2.94] pp |
| Realized flat return itself | [−83.08%, +7.54%] | [−26.13%, +30.03%] |

Only four months contain selected bets. These intervals are historical, nominal, and do not correct earlier repeated analysis or certify independence. A gap can be below −100 percentage points because predicted EV is subtracted from realized return; it is not a claim that the return itself fell below −100%.

**The evidence for exaggerated predicted EV is stronger here than the evidence that true long-run profitability is negative.** The latter return intervals include zero. Source: [return uncertainty](../data/08_reporting/production_market_calibration_ev5_20260926_210430/flat_return_uncertainty.csv) and the selection summary above.

We have not identified the architectural cause. Missing context, stale learned parameters and drift are hypotheses. Monthly decay-weighted refitting might help, but existing evidence does not demonstrate that it repairs this conditional error. Nor does a small seed/bootstrap spread rule out a shared bias across ensemble members.

**Bottom line:** the model's general predictive skill is not the same thing as a validated edge estimator. The immediate diagnosed issue is excessive confidence in the magnitude of selected model–market disagreements, amplified by odds and evaluated under an unproven early-information contract—not proof that every part of A0 is bad.

## Preservation and verification

No historical model, quote, selection, bootstrap result or previous research report was changed. No new financial strategy, gate sweep, training or deployment ran. This follow-up reads existing saved diagnostics and computes only transparent arithmetic interpretations. Exact source rows, source hashes and illustrative formulas are retained in `data/08_reporting/bookmaker_noise_research_20260927_200205/ev_48h_explanation_evidence.json`; its delivery receipt records preservation of the previous research bundle.
