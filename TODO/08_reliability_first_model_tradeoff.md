# 08 — Reliability-first model tradeoff

**Status: completed, 2026-10-01; no adoption.** Authorized by the user's request to create a goal/plan and execute the comparison after discussing worse log-loss versus more reliable probabilities. Accepted execution namespace: `reliability_tradeoff_20261001_092401_integrity_v2`; the initial attempt remains preserved separately. C6 failed the predeclared reliability-improvement gate. Annual Causal A0 remains the research reference; forward confirmation is **NOT_RUN, zero genuine future forecasts**.

## Goal

Determine whether a **fixed reduction in native neural training duration** improves an explicitly defined probability-reliability diagnostic while remaining within a small, predefined proper-score cost. Distinguish (1) a duration intervention, (2) a confidence-only transformation, and (3) retrospective model-selection optimism. Do not infer corrupted labels, per-match true probabilities, betting reliability or prospective calibration from this experiment.

This is a new historical-development comparison. It does not relax or rewrite Plan07's failed promotion gates, revive the exhausted betting loop, or authorize collection/deployment. Annual Causal A0, the canonical manifest, frozen prior artifacts and the operational engine stay unchanged.

## Fixed experiment

- **C0:** saved Plan07 native12 controls from `model_reliability_20261001_055321`; 2026 uses that run's pinned reference. Reuse shared rating/MLP experts, normalizers and all controls; no control refitting.
- **C6:** one native six-epoch fit per seed/origin, with the same architecture, initialization, ordered batches, optimizer, learning rate, target, auxiliary coefficient, gate RNG and data as C0. Seeds: `20260910`, `20260911`, `20260912`. Origins: January 1 of 2024, 2025 and 2026. Nine fits and **54 seed-epochs per execution**; no search, early-stopping selection or post-score variants. The integrity-only rerun and its additional computation are recorded below.
- A six-epoch call follows the first six epochs of C0's single twelve-epoch call. Check initial-state hashes and expected update counts. The earlier native6+12 arm resets optimizer/gate RNG and is **not** a pure eighteen-epoch duration control. No six-epoch endpoint was saved in that earlier arm.
- Preserve each origin's TRAIN/CAL/assessment membership. TRAIN ends before the previous July 1; CAL is the previous H2 with labels released before the origin. The three TRAIN/CAL/assessment counts are `27705/1379/4642`, `32625/1103/3805`, and `36205/1328/3103`.
- Fit only C6's native CAL beta parameters and constant mixtures, using C0's shared expert slopes and the existing recipe. This CAL is not cross-fitted. Never fit either arm on its assessment labels.
- Export all **11,550 canonical assessments**, all three modes (`full`, `no_w20`, `no_organization`), and raw-neural, calibrated-neural and assembled probabilities. Full-context assembled probabilities are primary. All modes must retain complete coverage; no-organization also removes W20 under the existing native contract.
- Score both endpoints on identical TRAIN and CAL rows as **in-sample diagnostics**, not forecast validation. Logged minibatch losses are not endpoint winner scores. Export endpoint winner probabilities and both raw and calibrated neural-ensemble scoreline likelihoods. Neither ensemble likelihood is the individual seed's stochastic training objective. Report per-origin values; pooled TRAIN rows repeat historical IDs across origins and are not unique matches.

## Confidence-only mechanism control

For each origin/mode, use assembled CAL forecasts to set

`s = sqrt(mean(logit(p_C6_CAL)^2) / mean(logit(p_C0_CAL)^2))`.

Apply `sigmoid(s * logit(p_C0))` without assessment labels. RMS logit magnitude is invariant to reversing individual A/B orientations; centered logit standard deviation is not. A scale above one is confidence matching, not softening. Zero C0 CAL dispersion is an explicit unavailable/error condition, not a fabricated fallback. This transformation is a diagnostic comparator only, not another candidate eligible for promotion. Positive scales preserve ranking within each origin/mode and preserve complementarity; differing annual scales can change pooled cross-year ranks. A zero candidate RMS gives a neutral, rank-collapsing diagnostic. Report C6's favorite and ordering changes separately.

## Reliability endpoint

1. Fix orientation and groups using **canonical annual A0**, never the candidate: A0 favorite and confidence intervals `[.5,.6)`, `[.6,.7)`, `[.7,.8)`, `[.8,.9)`, `[.9,1]`, crossed with calendar quarter. Numerical A0-neutral rows (`abs(p-.5) <= 1e-12`) are excluded only from favorite diagnostics, not proper scores or coverage.
2. For each cell `g` and model `m`, compute the mean residual `r_mg = mean(probability assigned to the fixed A0 favorite - fixed-favorite win indicator)`.
3. Define `G_m = sum_g (n_g/N) * r_mg^2` and `RMS_m = sqrt(G_m)`. Report signed gaps and counts for every cell, global fixed-confidence cells, quarter/year summaries and candidate-own favorite calibration separately.
4. The paired contrast `G_C6-G_C0` cancels the common empirical win-rate-squared term algebraically. It **does not cancel all finite-sample noise**, identify latent per-match probabilities, or prove general conditional calibration. Empirical squared gaps include variance; report that limitation.
5. Use 5,000 paired calendar-month block draws, seed `20261001`. Apply the same statistic to each resampled population: both arms share `n_g*/N*` weights within a replicate. An absent cell has zero population weight and an undefined cell mean, contributing zero—not an imputed calibrated cell. Report absent-cell and zero-total-support draw counts; never silently discard/redraw undefined aggregate draws. This is a population-mass-weighted functional, not a fixed-original-cell-weight bootstrap.
6. Report paired squared-gap and RMS-difference intervals. Proper-score contrasts use existing match-weighted and equal-month estimands. Repeat with source tournament-name × year blocks as an explicitly **uncertified edition proxy**. Report cluster counts, support and missing labels; do not silently pool names.

## Predeclared gates

Primary contrast: **C6 minus C0**, full-context assembled probabilities, all 11,550 rows. All conditions must pass for historical-development eligibility:

- Exact canonical coverage, ID/side alignment, finite probabilities and valid native inference contracts in every mode.
- **Reliability:** RMS point improvement at least `0.005` (0.5 percentage point) **and** the 95% monthly paired squared-gap interval has upper bound below zero. Directional evidence and point materiality are separate; this does not assert that the whole RMS interval exceeds 0.5 pp.
- **Global LL cost:** paired monthly 95% upper bound below `+0.005`. This borrows Plan07's profile guard tolerance, not its old global superiority gate.
- **Global Brier cost:** paired monthly 95% upper bound below `+0.001`. This is a newly chosen bounded research tolerance, not an inherited or financial threshold.
- **Critical LL guards:** the same `+0.005` upper-bound criterion for BO3, BO5, major events and source-unseen-player profiles, using the existing profile definitions. No outcome-picked profiles or exceptions.
- **Optional-context guards:** `no_w20` and `no_organization` assembled LL also satisfy the `+0.005` upper-bound criterion versus their corresponding C0 modes.
- **Temporal stability guard:** each year's quarter-cell RMS point worsening is at most `0.005`; report yearly intervals without presenting this point guard as a simultaneous-confidence guarantee.
- **Absolute calibration support:** existing own-favorite aggregate gap intervals for all/BO3/BO5/major/source-unseen-player are wholly within ±5 pp, with at least 200 nonneutral rows and six monthly blocks. Failure means tolerance **not established**, not automatically proven miscalibration.
- A C0-relative eligible candidate must additionally satisfy **the same reliability improvement criterion**, global LL/Brier cost and critical LL guards against canonical A0 before any **new research replacement eligibility** is recorded. Improving a weaker control is not a benefit over the retained reference. This safeguard is fixed before any new fitting/scoring. No deployment or prospective certification follows even if all gates pass.

Keep individual intervals and the all-required gate family distinct; profile tables are not simultaneous 95% confidence bands. ECE, tail counts, a few avoided upsets or sharper/softer probabilities alone cannot pass these gates. A negative/inconclusive result is a completed result, not a reason to change the threshold or try another epoch count.

## Selection-overfitting diagnostic

Use only C0 and C6, full context. Two fixed recipe selectors produce a separate retrospective sequence:

- 2024 uses C0 because no earlier assessment evidence exists.
- At the 2025 origin, use only released 2024 assessment labels; at the 2026 origin, use only released 2024–2025 assessment labels. Preserve the evidence-ID and release-cutoff ledger. Ties retain C0.
- **LL-first:** choose the lower earlier mean LL.
- **Reliability-first:** choose C6 only if the same C0-relative gates above pass on available earlier evidence; otherwise retain C0. The canonical-A0 replacement gate is not part of this descriptive two-arm selector.
- Compare realized policy forecasts on identical later rows. Separately show the whole-cohort hindsight LL winner as an explicitly optimistic descriptive comparator. This small fixed candidate set does not measure the project's entire historical selection bias or establish prospective superiority.
- Perturb current/future labels and release times outside an origin's admissible past; that origin's choice and evidence IDs must not change. Previously unavailable labels newly released before an origin are a different information scenario, not an invariance requirement.

## Interpretation and execution

1. Freeze the machine-readable protocol, exact source snapshot, input/control hashes, memberships and this goal before new fitting/scoring.
2. Run the nine C6 fits and export matched endpoint/assessment predictions. No post-score tuning, new architecture, quality mask, league exception, market head or betting simulation.
3. Run the existing benchmark entry point with the new reliability-study mode; reuse existing unclipped LL/Brier, coverage, profile and paired-block definitions.
4. Independently replay saved checkpoints and verify metrics, nonlinear calibration statistics, bootstrap intervals, scale optimality, selection chronology and hashes. Check BO1/3/5, side symmetry, cold/empty histories, nonmutation and a future-perturbation training/input case. Run focused regression once implementation is integrated, plus actual CLI smoke paths.
5. Publish the complete outcome, failed gates, TRAIN-versus-assessment patterns and mechanism-control result. Better TRAIN scores plus worse assessment scores can be consistent with duration overfitting; neither that pattern nor a calibration gain proves label noise or its cause.
6. Retain annual A0 unless the explicitly separate historical eligibility criteria pass. Seal any eligible recipe unchanged for genuinely new released inputs/outcomes; record **NOT_RUN, zero genuine future forecasts** here. Do not manufacture future evidence, roll dates, deploy, or automatically start another experiment.

Artifacts use `data/{05_model_input,06_models,07_model_output,08_reporting}/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/`. Preserve Plan07 and all rejected experiments. Update this plan, the existing research report/status and script index after execution; retain exact commands, checks and material limitations.

## Pre-assessment integrity correction

The initial `reliability_tradeoff_20261001_092401` execution was stopped before formal assessment after independent review found that the 2026 control redirect checked its completion hash but did not enforce the checkpoint/receipt hashes inside that completion. No changed real control artifact was established; the defect was a missing enforcement step. All nine finished candidate checkpoints, their **54 seed-epochs / 13,590 updates**, partial endpoint exports, original protocol/source ZIP and [retirement receipt](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401/retired_attempt.json) remain preserved and excluded from the decision.

Protocol v2 verifies and pins every redirected completion artifact before deserialization, at freeze, fit, load and evaluation. Regression cases prove that coherent checkpoint-plus-sidecar changes and receipt-only changes are rejected against the original completion. The unchanged nine-fit recipe was rerun in the new namespace: **108 total six-epoch candidate seed-epochs / 27,180 updates across the retired and accepted executions**, not eighteen distinct candidate recipes. All nine learned network/dependence parameter hashes match between executions; serialized checkpoint bytes do not. Two three-seed/one-epoch preflights (24 updates each) and two disposable one-update invariance fits are separate software diagnostics. Splits, seeds, objective, CAL policy, reliability endpoint, margins and all selection rules remain unchanged; no formal assessment outcome was used to choose the correction.

## Completed outcome and decision

Full-cohort proper scores cover all 11,550 canonical matches. Fixed-A0 quarter-cell calibration RMS uses 11,544 nonneutral rows, 55 cells and 33 monthly blocks.

| Forecast | Log loss | Brier | Calibration RMS | LL ≥ 2.5 |
|---|---:|---:|---:|---:|
| Annual A0 reference | 0.551246 | 0.187103 | 2.7773% | 33 |
| C0: native twelve epochs | 0.550351 | 0.186761 | 2.6172% | 47 |
| C6: native six epochs | 0.551218 | 0.187049 | 2.5741% | 53 |
| CAL-confidence-matched C0, diagnostic only | 0.550321 | 0.186751 | 2.6002% | 43 |

- **Primary rejection:** C6−C0 ΔLL **+0.000867**, monthly 95% interval **[+0.000153, +0.001661]**. The calibration RMS point improvement is only **0.0431 pp**, below the fixed **0.5 pp** criterion; its difference interval is **[−0.0972, +0.0311] pp**, and the squared-gap interval also crosses zero. All C0-relative proper-score cost, critical/optional-context, annual-stability and own-favorite absolute-support guards pass; they do not override the failed reliability gate.
- **Separate A0 safeguard:** C6−A0 ΔLL **−0.000028 [−0.001092, +0.000975]**. The 0.2032 pp RMS point improvement also fails the reliability criterion; BO5 LL interval upper bound **+0.005557** exceeds **+0.005**. Neither C6 nor the diagnostic confidence control is a new research replacement.
- **Confidence mechanism:** full-context CAL scales are **1.000116 / 0.992650 / 0.992447** for 2024/25/26. The control preserves within-origin/mode ranks and favorites, but differs across years. Its LL/RMS changes versus C0 are inconclusive. C6 changes favorites on **244 / 263 / 266** rows in full/no-W20/no-organization modes, so it is not merely the same ranking at another confidence scale.
- **Duration and selection:** twelve epochs have lower full-context winner LL than six on every origin's TRAIN, CAL and assessment for raw-neural, calibrated-neural and assembled forecasts. This dose comparison does not show the classic better-TRAIN/worse-assessment pattern; it does not establish absence of overfitting or identify label noise. Both chronological selectors choose C0 in all three years, using **0 / 4,642 / 8,447** earlier eligible rows; the hindsight best constant is also C0.
- **Uncertainty:** 5,000 paired monthly draws; 3,298 contain at least one absent cell, with zero undefined aggregate draws. The 444 source-name × year clusters are uncertified edition proxies. Empirical squared calibration gaps retain finite-sample variance, and individual profile intervals are not simultaneous guarantees.
- **Verification:** 43 focused tests passed, one existing warning. Real checkpoint replay covered **2,014,110 probabilities** within the inherited absolute tolerance `2e-6` (observed maximum `1.770589e-6`, not bitwise reproduction); **1,342,740** scoreline values matched an independent product oracle within `9.77e-15`. All nine direct C0 first-six loss prefixes, 182 artifact/source hashes, BO1/3/5 × three-mode inference, symmetry/cold/empty/nonmutation checks, selected-TRAIN future-input exclusion and six real-data selector perturbations passed. Independent arithmetic matched 184 headline/endpoint metric rows, 35 proper/gap intervals, eight nonlinear plus three annual calibration contrasts, 19 gate booleans, all **3,058 profile rows**, and nine confidence-rank checks. No full application-suite, prospective, financial or tournament certification is claimed.

Accepted evidence:

- [Frozen protocol](../data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json) and immutable `initial_goal.md`; the protocol's pre-execution `NOT_RUN` field is preserved, not a live execution status.
- [Assessment](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/assessment/evaluation.json), [decision](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/decision.json), and [forward boundary](../data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/forward_confirmation.json).
- Full interpretation and verification limits: [existing reliability report, §13](../reports/noisy_data_reliability_research_20260929.md#13-reliability-versus-log-loss-plan08-2026-10-01).

Commands actually executed from the repository root (existing frozen outputs intentionally refuse overwrite; these are the historical execution record, not an instruction to launch another run):

```bash
.venv/bin/python scripts/train_a0_reliability_tradeoff.py --freeze-run reliability_tradeoff_20261001_092401_integrity_v2
.venv/bin/python scripts/train_a0_reliability_tradeoff.py --protocol data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json --smoke
.venv/bin/python scripts/train_a0_reliability_tradeoff.py --protocol data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json
.venv/bin/python scripts/run_model_benchmark.py --reliability-tradeoff data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/protocol.json --output-dir data/08_reporting/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/assessment
.venv/bin/python -m pytest -q betting_app/tests/test_reliability_tradeoff.py betting_app/tests/test_a0_reliability.py betting_app/tests/test_a0_data_focus.py betting_app/tests/test_model_reliability_report.py betting_app/tests/test_model_benchmark.py betting_app/tests/test_research_benchmark_suite.py --junitxml=/tmp/reliability_tradeoff_20261001_092401_integrity_v2_final_regression.xml
```

**Final boundary:** retain annual A0, frozen prior models and the canonical manifest. No deployment, live collection, new market head, betting-loop restart or automatic next-duration experiment. Historical-development completion is not future confirmation.
