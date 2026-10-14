# What to change in the LoL model: targets, capacity, distillation and selected-bet reliability

**Subsequent authorized execution — 2026-10-04:** IDEA-028 is complete as `native_capacity_20261004_01` (**NOT_ADOPTED**); IDEA-029 was **SKIPPED_BY_FROZEN_EVALUATION_GATE**, with zero KD fits because the W16/E12 major/C0 noninferiority contingency failed. Nine continuous24-epoch trajectories and18endpoints were actually trained. All three multiplicity-adjusted primary LL intervals cross zero; lower point loss does not establish a replacement. [Execution results and verification](../docs/RESEARCH.md), [sealed decision](../data/08_reporting/a0_capacity/native_capacity_20261004_01/decision.json). The research-only findings below remain the original study, not a new or unfinished execution queue.


Research date: **2026-10-04**. Status: **completed research and saved-forecast diagnostics; no new model training or adoption**.

Evidence run: `model_redesign_research_20261004_01`. [Frozen diagnostic protocol](../data/05_model_input/model_redesign_research/model_redesign_research_20261004_01/protocol.json); [numerical verification](../data/08_reporting/model_redesign_research/model_redesign_research_20261004_01/independent_verification.json); [source/access ledger](../data/08_reporting/model_redesign_research/model_redesign_research_20261004_01/source_ledger.json).

## 1. Executive decision

**Do not replace the win-probability objective with a new proxy or launch another broad objective sweep. First measure the native model's capacity/optimization tradeoff. Treat distillation as a controlled compression/regularization experiment, not a way to manufacture a betting edge.**

The most consequential demonstrated problem is the jump from *better average sports probabilities* to *reliable positive expected value at the prices we select*. The newest connector improves the older C0 on its own selected fixtures, but still forecasts those fixtures materially worse than the same-cutoff Pinnacle reference. The corresponding difference outside its selections is inconclusive. This is a selection-conditional forecasting failure in an archived scenario, not a diagnosis that every probability is too confident or that the network is too large.

| Question | Evidence-backed answer | Decision |
|---|---|---|
| Are we learning the wrong outcome? | A win label is the correct observation for a win-probability forecast. The native neural expert already learns terminal scorelines plus masked player-performance targets. | Retain probabilistic winner/scoreline semantics; do not relabel legitimate upsets as bad data. |
| Would scorelines help instead? | Already tested. A later matched 18-fit native study favors terminal training over winner-only training by a small historical amount. | Preserve that result; scoreline supervision is not an untried improvement. |
| Is the network too big? | Actual C0 checkpoints contain **6,606 trainable values per history member**, including its auxiliary head and eta. Three members plus four other subnetworks are not one 6k model. | No evidence establishes oversizing. Count is an inventory, not a generalization theorem. |
| Is it too small? | Twelve epochs beat six in the completed comparison. No matched native width experiment or per-epoch held-out curve identifies a capacity bottleneck. | A bounded width-by-duration experiment is the highest-value next modeling diagnostic. |
| Could a smaller distilled model be better? | Yes in principle and in published experiments; teacher bias, capacity mismatch and temporal shift can prevent it. | Use a matched small-student control and a training-only teacher. Measure probabilities and cost separately. |
| Should we run more experiments? | We have already run many target, weighting, calibration, architecture and financial variants on exposed history. | Fewer hypothesis-discriminating experiments, explicit stop rules, no automatic search restart. |
| What would establish betting reliability? | Decision-time evidence and selection-conditional validation, not an attractive global LL, ECE, teacher-agreement score or bankroll example. | Keep financial certification separate from continued historical sports development. |

The recommended sequence is **one native capacity study**, then **one conditional native-expert distillation study** if compression or regularization is still a concrete objective. Both are proposals, not executed work or authorization to train. Canonical annual A0, the C0/connector artifacts, production EXP081/hybrid, the manifest and the exhausted 10/10 journal remain unchanged.

## 2. Method, scope and model identities

### 2.1 What this investigation actually did

Five parallel literature tracks covered targets, capacity, distillation, selection-conditioned reliability, and sports representations. A sixth, read-only code track traced the native implementation. Parent verification then followed the inheritance chain, loaded hash-verified local checkpoints in an isolated CPU process, counted tensors, read completed experiment receipts, and calculated new diagnostics from already saved predictions. The literature search includes an October 1, 2026 tabular-distillation preprint, not only older neural-network papers.

The bibliography contains **36 external sources**, including methodological research, empirical papers, preprints and a learning-curve review. Thirty had relevant full-text sections inspected; six are conservatively marked abstract/publisher-summary-only. An accessible full text is not independent replication or proof of applicability to professional pre-draft LoL. The newest company-authored tabular benchmark is explicitly treated as a preprint, not external validation of this project.

No network, calibrator, connector weight, selection threshold or financial policy was fitted. No neural forward replay, source acquisition, deployment, live database access or tournament simulation ran. Checkpoint deserialization/counting is not a new prediction replay. The new bootstrap calculations are descriptive analyses of exposed historical data, with a protocol frozen before those calculations but after earlier research had already exposed the outcomes.

### 2.2 The identities must stay separate

| Identity | Meaning here |
|---|---|
| Canonical annual Causal A0 | The research benchmark reference in `conf/base/research_benchmark.json`; not the live application model. |
| Native12 C0 | The assembled control recipe, including the 2026 `a0_data_focus_20260929_192443/control` checkpoints. Its TRAIN/CAL/assembly recipe differs from canonical annual A0. |
| New C0 connector | `c0_raw_connector_20261002_033658`, saved column **`p_c0_connector`**. This is the model diagnosed below. |
| Connector03 | The earlier A0-anchored connector, not the newest C0-anchored candidate. |
| Operational EXP081/hybrid | A separate application identity. No inference about its current live predictions or deployment follows from this study. |

The [Plan09 execution report](plan09_c0_connector_and_forward_evidence_20261002.md) records three passing primary sports contrasts for the newest connector, but two unmet annual-A0 noninferiority guards: BO5 upper LL difference **0.005183914** and source-unseen **0.005193668**, against a **0.005** margin. Its decision is **NOT_ADOPTED**. Failure to establish the required noninferiority is not proof of harm. A later descriptive betting diagnostic does not reverse that decision.

## 3. What is demonstrably going wrong

### 3.1 Average probability quality does not certify selected value

A model can be calibrated relative to its own probability while being wrong conditional on additional information used to select actions. For betting, that additional information includes prices, counterpart probabilities, the chosen side, and the selection rule. Conditional-calibration and conditional-predictive-ability theory make the distinction explicit.[3,4] Decision-calibration work likewise concerns a specified class of decisions and assumptions; it does not supply an automatic guarantee for arbitrary future betting strategies.[34]

The optimizer's curse gives a second relevant mechanism: selecting the largest noisy estimated benefits can make selected benefits systematically optimistic.[5] That mechanism is plausible here, but this research does **not** identify how much of the observed gap comes from selection noise, missing sports information, changing distributions, model misspecification or source limitations. The arithmetic below is not a causal attribution.

### 3.2 New same-fixture evidence on the current model's own selections

The accepted v2 Pinnacle scenario contains **1,615 fixtures**; the original **1,619-opportunity** inventory remains the bootstrap universe. The current connector's saved raw-net-EV>5% policy selects **330**; **1,285** are unselected. These are not the earlier 328 A0-selected fixtures, nor the older prior-roster cohorts. Four opportunities excluded by the already completed clock-order correction are not silently restored.

All three forecasts below are scored in the same A/B orientation on exactly the same fixtures. Log loss is **unclipped**; supplied probabilities are strictly between zero and one.

| Fixed subset | N | New connector LL | C0 LL | Pinnacle reference LL |
|---|---:|---:|---:|---:|
| All accepted fixtures | 1,615 | 0.574065 | 0.574609 | 0.564865 |
| Current connector selected | 330 | 0.627486 | 0.631840 | 0.578499 |
| Unselected | 1,285 | 0.560346 | 0.559912 | 0.561363 |

Paired monthly-block intervals use **5,000 draws**, seed **20261004**, and the complete **nine-month** opportunity universe:

| Contrast | Match-weighted ΔLL | 95% monthly interval | Equal-active-month point |
|---|---:|---:|---:|
| All: new−C0 | −0.000544 | [−0.001572, +0.000376] | −0.001178 |
| All: new−market | +0.009201 | [−0.002447, +0.021036] | +0.012498 |
| Selected: new−C0 | **−0.004355** | **[−0.007250, −0.001409]** | −0.004229 |
| Selected: new−market | **+0.048986** | **[+0.004253, +0.099937]** | +0.066264 |
| Unselected: new−C0 | +0.000434 | [−0.000801, +0.002036] | +0.000162 |
| Unselected: new−market | −0.001017 | [−0.009354, +0.004917] | −0.006554 |

Sensitivity using **111 source-tournament-name/year proxies** gives selected new−market interval **[+0.004499,+0.098337]** and selected new−C0 **[−0.008200,−0.000726]**. These labels are not certified independent tournament editions. Equal-proxy points are respectively **+0.095835** and **−0.005596**. Only equal-block **point estimates**, not equal-block bootstrap intervals, were newly calculated in this run.

**Interpretation:** the connector helps on the selections relative to C0; the selected forecasts nevertheless remain worse than the market reference in these descriptive comparisons. The unselected difference does not establish equivalence. The all-fixture difference does not establish universal market superiority. These are post-development conditional diagnostics, not independent discoveries or a justification for hardcoding a new exclusion rule.

Fixed disagreement profiles reinforce where to inspect, not what to remove:

| Absolute new-model/market probability gap | N | Selected N | New LL | Market LL |
|---|---:|---:|---:|---:|
| <5 pp | 719 | 11 | 0.552376 | 0.554742 |
| 5–10 pp | 483 | 62 | 0.552419 | 0.550718 |
| 10–20 pp | 327 | 172 | 0.617555 | 0.591831 |
| ≥20 pp | 86 | 85 | 0.711609 | 0.626409 |

Nonneutral favorite agreement covers 1,430 fixtures; disagreement covers 177. Eight numerically neutral rows are retained in the main cohort and excluded only from that favorite profile. No outcome-selected league or probability threshold was introduced.

![Conditional proper-score and EV diagnostics](../data/08_reporting/model_redesign_research/model_redesign_research_20261004_01/conditional_diagnostics.png)

### 3.3 The selected EV error is much larger than the measured sports improvement

On those same 330 selections, the connector predicts **154.08 wins**, the no-vig Pinnacle reference **101.01**, and **109** occur. The market is not being declared truth: its selected residual has substantial uncertainty.

With decimal odds O, tax fraction τ=0.12, chosen-side probability p and result y:

$$\widehat{EV}=(1-\tau)Op-1,\qquad R=(1-\tau)Oy-1.$$

For the same chosen-side market reference q, the exact identity is

$$\widehat{EV}-R=(1-\tau)O(p-q)+(1-\tau)O(q-y).$$

The first term is an odds-weighted forecast disagreement; the second is a market residual. Neither is an identified cause.

| Fixed selections | Mean predicted EV | Mean realized unit return | Overstatement | Model−market component | Market residual |
|---|---:|---:|---:|---:|---:|
| All 330 | +35.9511% | −9.0099% | **44.9609 pp** | +53.1604 pp | −8.1995 pp |
| Own (5%,10%] band: 76 | +7.5308% | −15.0395% | **22.5703 pp** | +24.3021 pp | −1.7318 pp |

The newly calculated monthly overstatement intervals are **[26.3294,66.2818] pp** for all 330 and **[2.4576,39.8437] pp** for the 76 near-band selections. The all-selected market-residual interval **[−27.8907,+12.5147] pp** does not establish market calibration. Slight differences from the prior betting report's endpoints reflect the separately frozen bootstrap seed, not changed bets or probabilities; both receipts remain intact.

The completed [betting results](../data/08_reporting/betting_ev_reliability/c0_betting_20261002_085551/results.json) separately report flat-stake Pinnacle ROI **−9.01%** and capped-quarter-Kelly ROI **−6.81%**, with Kelly maximum drawdown **68.18%**. Those ledgers were **not rerun here**. Monetary cent rounding makes their flat ROI slightly different from the unit-return mean above. Removing tax on the same saved selections had a positive but highly uncertain point ROI, not established profitability. Better bookmaker shopping also cannot be equated with correcting probabilities; prior price-conflict sensitivity materially changed the result.

Consequently, neither a small global ΔLL nor a smaller model is evidence that a displayed +5% EV has become trustworthy. Selection and pricing have to be assessed explicitly. Sports-forecast research can continue without waiting for that financial certification.

### 3.4 Three process mistakes to stop

1. **Treating a global score as a decision guarantee.** Keep full-cohort LL/Brier, fixed-selection probability errors and own-policy financial outcomes as separate estimands. Fixed IDs/sides/prices/stakes do not change realized flat PnL when only the model probability changes.
2. **Re-proposing completed experiments.** The archive contains a broad target/H matrix and a stronger native winner-versus-terminal study than a brief handoff alone suggests. Read the actual completed receipts before calling an idea new.
3. **Treating repeated historical improvement as independent confirmation.** A chronological split prevents certain temporal leaks; it does not undo years of adaptive hypothesis selection on its outcomes.[6] Development intervals remain useful diagnostics, but are not a reset of the research history.

## 4. Is the outcome the wrong target?

### 4.1 The target must match the question, not suppress legitimate randomness

If the product promises a series-win probability, its object is

$$p^*(x)=P(Y=1\mid X=x,\ \text{information available at the forecast origin}).$$

For a Bernoulli outcome, expected log loss is uniquely minimized by the true conditional probability. Brier has the same unrestricted population optimum. A team losing once with true win probability 0.8 is not a corrupted training label; losses are part of the distribution being predicted.[1] Paired-outcome models remain principled sports baselines, not obsolete merely because their observed labels are binary.[7]

Distinguish three issues:

- **Outcome randomness:** unpredictable map events, execution and other variation remaining after conditioning. More data can estimate its distribution; relabeling an upset does not remove it.
- **Estimation/model uncertainty:** limited historical support, a misspecified representation, optimization, calibration estimation and distribution change. These can be reduced, but no single mechanism has been identified here.
- **Unknown decision-time context:** roster, competition unit, future draft and map side. These must be represented, marginalized or withheld under the information contract, not read from the future and called pre-match knowledge.

The scientific objective is sharp forecasts subject to calibration, not confidence reduction for its own sake.[2] ECE can reward an uninformative near-constant forecast; it cannot replace LL, Brier, discrimination, support and application-specific reliability checks.

### 4.2 The native model already has richer supervision

The native neural training loss is:

$$L=L_{\text{terminal Beta scoreline}}+0.1L_{\text{masked auxiliary SmoothL1}}.$$

Its main likelihood is over the **legal terminal series result**, not independent duplicated series labels masquerading as maps. A learned dependence parameter enters the Beta terminal likelihood. The auxiliary head predicts **12 player-performance channels**; missing observations are masked and loss is normalized within the existing player/series contract. The auxiliary loss is SmoothL1/Huber, not ordinary MSE.

The actual hierarchy is `NonlinearHistoryModel → BetaTrainedModel`, which constructs `CoherentModel/CommonMapResidual`. The inherited final layer has **one shared map-logit output**, not three BO-specific skill outputs. BO enters the terminal-series likelihood/projection. `TanhHistoryEncoder` adds a parameter-free nonlinearity before attention. Descriptions based only on the generic parent class get this wrong.

Therefore “replace binary training with richer targets” misdescribes the current native recipe. The separate raw specialists and historical application models have their own objectives; this statement is about the native history expert, not every model in the repository.

### 4.3 The decisive local objective comparison was already completed

The [native paired-objective artifact](../data/08_reporting/a0_target_experiment/ev_all_approaches_20260927_220902/paired_objective_results.json) records **18 real neural fits**, three origins × three seeds × two main objectives. Architecture, shared experts and terminal-distribution calibration policy were matched; only the neural TRAIN main objective differed.

On all **11,550** canonical rows:

- **Winner minus terminal ΔLL = +0.000446114**, monthly 95% interval **[+0.000011263,+0.000902319]**.
- ΔBrier **+0.000169186**, interval **[+0.000011242,+0.000331396]**.
- Equal-month ΔLL **+0.001141708**; 33 monthly blocks and 5,000 draws.
- Winner-only LL is worse pointwise in 2024, 2025 and 2026.
- The 444 source-name/year sensitivity also has a positive LL interval; those are not verified editions.

This artifact uses its recorded epsilon **1e−12** policy; the new conditional diagnostics above use unclipped probabilities. The terminal branch nearly reproduces the frozen reference, but its recorded maximum probability difference **2.07e−5** is not byte-exact reproduction. Do not silently turn this into an exact current-C0 ablation.

An earlier coherent-family comparison had **scoreline minus winner ΔLL −0.000184 [−0.000645,+0.000308]**, an inconclusive result under a different recipe. These findings coexist: the later matched native experiment is evidence for preserving its terminal objective, not a theorem that richer labels always win or a prospective confirmation.

### 4.4 Continuous process labels can help—and can be the wrong proxy

Joint outcome/process sports models can share information through latent strengths. Some sports studies find gains from particular additional responses; others find similar winner performance from direct outcome and score-based models.[9,10] The relevant question is not “is this label continuous?” but “does this response teach a representation that improves the future target under the available information set?”

Player statistics mix strength, opponent, role, game state, strategy, duration and selection of who actually played. Hierarchical sports research illustrates why context and unequal participation matter.[31,32] These are transferable modeling considerations, not evidence that importing a soccer xG target or basketball random effect will improve LoL.

The repository already supplies observed performance targets. Its matched reconstruction experiment improved reconstruction loss while worsening final forecasts versus its matched-update control: **ΔLL +0.001411 [+.000106,+.002930]** on exposed 2026. That is direct evidence against the tested pretext recipe, not proof that all auxiliary learning is useless.

Auxiliary-task research also shows that negative gradient cosine is not sufficient evidence of harmful transfer.[12] Measuring gradient norms or conflicts could inform a future diagnosis, but would not by itself justify deleting the auxiliary loss, applying gradient surgery or starting a loss-weight grid. Preserve the earlier positive auxiliary-target findings and the later negative reconstruction result.

Training-only future performance can be a legitimate label or privileged teacher signal, provided all those labels were released before the training lock. It cannot be used as a feature at an earlier prediction origin.[11] Training on post-match supervision and leaking post-match information into inference are different operations.

### 4.5 Profit, return-MSE, focal and CLOSE targets are not untouched alternatives

The completed `ev_all_approaches_20260927_220902_matrix_v5` study already included **19 monthly head types and 2,014 head receipts**: sports/market calibration, H log loss/Brier, return-MSE, residual, weighted/focal/smoothed variants with corrections, ranking, CLOSE/CLV, direct-profit and Kelly objectives. These counts are computation records, not independent observations or 2,014 distinct scientific hypotheses.

On the same **2,278** eligible archival OPEN fixtures, H failed both primary predictive controls:

- H−M-cal ΔLL **−0.000416 [−0.001083,+0.000630]**.
- H−M-shape **−0.000192 [−0.000866,+0.000837]**.

No policy cleared the primary positive lower-bound financial increment against M-cal. Some historical points were favorable, including a fixed blend, but the study did not establish executable financial superiority. Preserve its old timing/support limitations and do not conflate it with the separate invalid `run005`.

The newer **strict G0-admitted H/control protocol is still NOT_RUN**. This means *that stricter evidence regime* has not been executed—not that market heads or profit targets have never been fitted. A rerun requires materially different admitted evidence and a new bounded hypothesis, not a renamed sweep.

There is also a useful exact identity. If predicted return is parameterized through p,

$$(\widehat R-R)^2=((1-\tau)O)^2(p-y)^2.$$

Thus return-MSE is odds-squared-weighted Brier, not a new observation of latent “true skill.” It emphasizes high-odds errors. Positive outcome-independent weights preserve the unrestricted conditional proper-loss optimum when the complete conditioning information is modeled; with informative omitted odds or restricted capacity, weighting can change the best fitted estimand. A checked two-group example moves the optimum from 0.50 to 0.74 when weights are 1 and 9 and group probabilities are 0.2 and 0.8.

Decision-focused learning is a real research field: useful gains have been demonstrated in specified optimization settings.[13,14] It does not follow that a profit-trained output is a calibrated probability, that it transfers to tournament simulation, or that another financial sweep is justified after the completed local matrix. Reade and colleagues likewise provide a sports example where better scoreline information did not deliver consistent betting returns.[8]

## 5. Is the native model too large or too small?

### 5.1 Actual checkpoint inventory, not a name-based guess

All three 2026 C0 history checkpoints were deserialized in a fresh CPU process after checkpoint/receipt-contract and archived-source hash checks. No fit or forward pass was performed. [The tensor inventory](../data/08_reporting/model_redesign_research/model_redesign_research_20261004_01/native_capacity_inventory.json) records every parameter shape.

| Component, one native history member | Trainable values |
|---|---:|
| Player 21→32 affine | 704 |
| History 33→32 affine, followed by tanh | 1,088 |
| Learned empty-history token | 32 |
| Scalar attention 32→1 | 33 |
| Shared utility hidden layer 134→32 | 4,320 |
| Shared map output 32→1, no bias | 32 |
| Auxiliary 32→12 affine | 396 |
| **Network** | **6,605** |
| Learned eta | 1 |
| **Total per member** | **6,606** |

Three history members contain **19,818** such values. The saved rating subnetwork has **250** network parameters, and each of three full-MLP subnetworks has **7,179**. The seven-network-plus-eta subtotal is **41,605**. This excludes separately fitted rating offsets/state, preprocessing, CAL parameters and the raw-trio specialist added by the connector; it includes auxiliary heads needed during training. It is **not a count of every fitted quantity or the complete system's memory footprint**.

A preliminary source-only count of 6,670 missed the coherent subclass's replacement of the generic 32→3 output with 32→1. The saved tensors and complete inheritance trace both give 6,606. This was a research-description correction, not discovery of an implementation bug.

The native encoder is attention pooling over each player's last **16 maps**, not a large Transformer. It combines player state, nonlinear history summaries and team-level mean/dispersion with optional organization/W20 context. Roles, age and opponent information are already represented. It would be wrong to infer that mean pooling means roles are absent, or to propose age, ratings or opponent context as entirely new information.

### 5.2 What the training evidence does and does not establish

The 2026 control uses **36,205 TRAIN** series before July 2025 and **1,328 CAL** series from 2025H2 with released labels before 2026. Each member receives **3,396 optimizer updates** over 12 epochs at batch size 128. The many overlapping player/history tokens and auxiliary cells are not independent extra match labels.

For seed20260910, recorded TRAIN main NLL falls **0.836104→0.820829**, and auxiliary loss **0.305570→0.282548**. The other two seeds show similar declining training trajectories. The receipts do not supply a per-epoch held-out generalization curve.

![Recorded training curves, not held-out curves](../data/08_reporting/model_redesign_research/model_redesign_research_20261004_01/native_training_curves.png)

Plan08 found native6−C0 ΔLL **+0.000866935 [+0.000153092,+0.001660772]** on 11,550 rows. Subsequent endpoint audit records **18/18** raw-neural/full-assembly × TRAIN/CAL/ASSESS × origin comparisons favoring twelve epochs. Six epochs is less training, not a smaller architecture. The earlier native6+12 finding failed its designated earlier-origin replication; it cannot be revived by selecting the exposed2026 result.

This evidence argues against the specific story “twelve epochs obviously overfit relative to six.” It does **not** establish that 24 epochs will help, that the model has insufficient capacity, or that the current training objective is optimized. Winner LL and the actual terminal-NLL-plus-auxiliary SGD objective are different quantities.

### 5.3 Capacity is a function of architecture, optimization and data—not a parameter/row ratio

There is no defensible universal rule that 6,606 parameters are too many or too few for 36,205 correlated series. Learning-curve research distinguishes performance versus sample size from a training curve versus epochs and documents diverse curve shapes.[23] Double-descent experiments show why monotone intuitions about width and training time can fail, but do not establish that this model is in a double-descent regime.[24]

A useful diagnostic has to separate:

- **Representational limitation:** larger width reduces appropriately measured training loss and improves later probabilities under a matched recipe.
- **Optimization limitation:** additional updates at the same width improve later probabilities; simply widening at a fixed optimizer configuration may or may not help.
- **Variance/overfitting:** a larger or longer-trained model fits TRAIN better while later generalization deteriorates under matched information and calibration.
- **Information limitation or shift:** width and duration fail to improve the relevant later errors, while missing context or changed relationships remain plausible. Failure of a small grid would not prove an irreducible Bayes-error ceiling.

These are diagnostic patterns, not single-metric causal proofs. In particular, lower TRAIN loss plus flat held-out LL can arise from several mechanisms.

### 5.4 Bigger tabular models are not an automatic escape

TabReD finds that temporal rather than random splits change model rankings, and that simple MLP-like and boosted-tree methods remain strong in its industrial datasets.[25] TableShift similarly separates in-distribution performance from robustness to distribution change.[28] A 2025 TabPFN finetuning study reports strong IID results but less stable performance on richer temporally shifted datasets.[35] Older tabular comparisons provide another reason to retain strong tree/simple-network controls rather than judging models by novelty.[29]

None of this proves that a tree is best for LoL; the project already contains strong tree/raw-specialist comparisons. It does mean that “replace it with a foundation model” is an architecture hypothesis, not a literature-backed solution to these betting errors.

One ICML temporal-tabular paper reports benefits from a changed splitting protocol.[26] That is not permission to mix future labels across a forecast origin here. Randomization inside an already closed TRAIN partition is different from random future/past evaluation. Context-driven residual-shift research supports looking for persistent conditional errors, but it does not justify outcome-picked leagues or automatic context adapters.[27]

## 6. Distillation: useful experiment, wrong default explanation

### 6.1 What distillation can reasonably achieve

Distillation trains a student to approximate teacher outputs. Its original motivation includes transferring an expensive ensemble into a cheaper function.[15,16] Published born-again experiments also show that a student need not be smaller and can sometimes outperform its teacher.[17] Therefore the correct answer to “could a smaller student improve?” is **possibly**, not “impossible” and not “it removes the noise.”

Potential mechanisms include a smoother optimization target, ensemble averaging encoded in the teacher, an inductive bias that discards unhelpful variation, and a smaller function that generalizes better. None has yet been measured for this native model. A small checkpoint count also does not identify the serving bottleneck: feature construction, rating/history state, shared experts and the tournament workload have not been profiled here.

The newest directly relevant preprint, **Jeong et al., October1,2026**, studies TabICLv2/TabPFNv3 teachers and TabM/XGBoost students. It reports matched student improvements and median inference speedups of **3.0–21.6×**.[22] Important qualifications:

- The teachers and datasets are not this sports model or professional LoL.
- TALENT binary classification is evaluated using **1−ROC-AUC**, not binary log loss or selected-EV calibration.
- Their recipe uses full-context teacher predictions and synthetic query generation; that is not automatically valid for linked roster/history tensors.
- Generic per-column CutMix could create impossible player/history/context combinations here. The proposed native study below uses real causal rows only.
- The paper is a new company-authored preprint, with no independent replication established in this investigation.

This is evidence that distillation deserves a controlled test, not a promised improvement in probabilities. ScoringBench's 2026 regression abstract independently illustrates that point-estimate and probabilistic scoring can rank models differently; it is supporting evaluation context, not binary-LoL evidence.[36]

### 6.2 A biased teacher remains a biased target

For a fixed binary teacher probability t(x), true conditional win probability r(x), and unrestricted population minimization at temperature1,

$$L(s)=(1-\lambda)\operatorname{CE}(r,s)+\lambda\operatorname{CE}(t,s)
\quad\Longrightarrow\quad
s^*(x)=(1-\lambda)r(x)+\lambda t(x).$$

If r=0.70, t=0.80 and λ=0.5, the optimum is 0.75, not 0.70. Pure distillation copies the teacher optimum. This identity does not predict every finite-sample student's behavior; regularization and restricted capacity may make a student better or worse. It establishes why a teacher's conditional bias is not automatically cured by compression. Theory on imperfect teachers and empirical capacity-mismatch work support that caution.[18,19]

Probability calibration is a separate objective from accuracy.[20] A 2025 computer-vision study reports benefits from teacher calibration in KD, but it does not establish a universal calibration benefit or authorize refitting a teacher on assessment outcomes.[21] In this project, a teacher that overstates selected value can transfer that error.

### 6.3 Ordinary in-TRAIN supervision is not automatically leakage

A teacher trained on the same **closed TRAIN** examples as its student can legitimately generate targets for them; ordinary KD explicitly permits this.[15] The 2026 tabular preprint even prefers full-context supervision in its benchmarks.[22] Out-of-fold teacher outputs can be a useful control against memorization, but are not a universal requirement for every distillation design.

The real boundaries are the **complete recipe and outer partitions**:

- Teacher weights, calibration, mixture fitting and any recipe selection used to produce student TRAIN targets must not use the student's outer CAL or assessment labels.
- A final C0 teacher calibrated on 2025H2 cannot supply student TRAIN targets while that same 2025H2 is claimed to be untouched outer CAL. The leakage is through teacher calibration/assembly, even if all dates precede the 2026 assessment.
- A later teacher may label earlier closed-TRAIN rows retrospectively when every teacher dependency stays inside that TRAIN lock; those targets are not historical as-of forecasts.
- A teacher using later CLOSE, draft or other privileged information must still be trained/admitted under a distinct contract. At inference, a student can recover only relationships available through its own inputs, not arbitrary hidden realization-specific information.[11]

This is why the proposed first KD study compresses the **training-only native neural group**, not an unexamined final assembled teacher.

### 6.4 Winner-only compression loses information needed for richer outputs

A winner probability does not identify a terminal-scoreline distribution. For a BO3, the two distributions over `(A2-0,A2-1,B2-0,B2-1)`

$$(0.4,0.2,0.2,0.2),\qquad(0.1,0.5,0.3,0.1)$$

both give P(A wins)=0.6, but imply expected lengths 2.4 and 2.6 maps. The counterexample was checked numerically.

The native neural family has terminal-distribution semantics. The **full assembled C0/connector's current public winner output is not a certified full joint-scoreline teacher**. Do not invent one by projecting a series probability through BO again, or assume a distilled winner head certifies tournament placements and scorelines. Plan09 explicitly leaves an eligible joint-scoreline adapter and official-edition evidence unresolved.

A winner-only student remains a possible separately scoped experiment; it would need its own direct-label winner control and could not inherit joint-output certification. The preferred bounded proposal below keeps the native terminal distribution for the neural component only.

## 7. Information and sports representation: what remains plausible

The model already has player ratings, experience/inactivity, roster coplay, optional organization/W20, recent map tokens, role, statistics, age and opponent context. Renaming any of these “dynamic strength” does not add information. Dynamic latent-strength research is useful for comparing update/representation assumptions, not proof that the existing ratings should be replaced.[30]

There are plausible structural limits: a short finite history, scalar attention pooling, and compressed team summaries may omit some interactions or adapt poorly to change. These are **[INFERENCE] candidate mechanisms**, not diagnosed causes. The native token encoder is already nonlinear; proposing its first tanh layer would duplicate existing functionality. Prior opponent-conditioned retrieval, player-pair/GNN work, and roster-form corrections also cannot be advertised as wholly new.

The completed FamilyMapBridge study provides a useful example of honest separation: it improved its matched player-rating control, yet integration into C0 failed its designated gates. A positive standalone component result is not evidence that another full-model mixture or representation will help. Likewise, input-fusion/separate-encoder/learned-connector architectures were tested; their outcomes must remain in the comparison register.

More detailed observations may eventually be more valuable than more parameters, but a proposed field must pass three checks before becoming the next model feature:

1. It represents information not already supplied or reconstructed by the current input contract.
2. Its source, player/competition identity and release time can be established at the intended origin, with missingness preserved.
3. A specified failure mechanism predicts an out-of-time improvement beyond a matched existing-model control; a few selected losses are not the justification.

The direct LoL champion-experience paper studies already selected champions in NA ranked play.[33] Its accuracy cannot be imported into professional pre-draft forecasting, where future draft is unavailable. The same warning applies to suggested side, scrim, motivation or patch explanations without source evidence. No specific missing sports factor was established by this investigation.

## 8. A finite experiment program, not another sweep

### 8.1 Experiment A — native capacity × continuous duration

**Priority: first. Status: proposed, not run.** Its purpose is to distinguish whether native width or additional continuous optimization helps under the existing information and objective. It is not another six-epoch reliability attempt, another mask recipe, or a new architecture family.

**Fixed design**

- Use the established 2026 development split: 36,205 pre-July2025 TRAIN rows; 1,328 released 2025H2 CAL rows; all 3,103 canonical2026 assessment rows through September8. All remain exposed historical development. No market-based TRAIN eligibility.
- Widths **16,32,64**, changing corresponding player/history/utility/auxiliary dimensions consistently while preserving inputs, pooling semantics, one map-logit head, eta, terminal likelihood, auxiliary coefficient0.1, optional modes and symmetry.
- Formula for this specific shape-preserving width change: `2*h*h + 142*h + 14`, including auxiliary head and eta. Proposed per-member counts: **2,798 / 6,606 / 17,294**. These are architectural calculations, not trained models.
- Three fixed paired seeds **20260910/11/12** and the same TRAIN order, gating RNG policy, batch size and optimizer recipe. Seeds pair repeated runs; different widths cannot share identically shaped initial weights.
- **Nine training trajectories** to epoch24, with prespecified checkpoints at12 and24: six width/duration cells, eighteen seed checkpoints. No restart at epoch12; preserve optimizer/RNG state within each trajectory. Reproduce the width32/epoch12 control before interpreting other cells. Do not assume old checkpoints include resumable optimizer state.
- Keep rating/MLP experts and input normalizers fixed. Fit each neural calibration/assembly by the same past-only CAL rule; report raw/calibrated neural winner and scoreline results, and assembled winner results, separately. Shared CAL is a fitting set, not an unbiased report of calibrated generalization.
- Record the actual main loss, auxiliary loss, eta, gradient diagnostics, update counts and TRAIN curves. Also record prespecified per-epoch raw held-out diagnostics; they are observational curves, not a license to select extra epochs after seeing assessment results.

**Three primary contrasts, fixed before fitting**

1. Width64−width32 at12 epochs: does more capacity help under the native update budget?
2. Width16−width32 at12 epochs: does lower capacity help or retain adequate quality?
3. Width32/24−width32/12: does continued optimization help?

The other fixed cell contrasts characterize interactions; they are not a post-hoc winner search. Same-update comparisons are not same-FLOP comparisons. A fixed learning rate is a deliberate test of the native recipe, not proof of each architecture's fully tuned optimum.

**Decision rule**

Use paired monthly blocks, at least5,000 draws, match-weighted and equal-month estimates, and source-name/year sensitivity. For a development claim on any of the three primary LL contrasts, use a conservative **98.333% per-contrast interval** rather than silently treating three independent95% passes as one test. This is a within-study multiplicity convention, not a cure for historical adaptive reuse or a rigorous finite-sample bootstrap guarantee.

A candidate must also have nonworse Brier point, full required coverage, and satisfy the existing **+0.005 LL noninferiority guards** against C0 and annual A0 for BO3/BO5/major/source-unseen; inadequate support is not a pass. Report fixed current-selection errors diagnostically without fitting to the selected330 or changing the policy. Do not call a sports pass financial adoption.

If no fixed candidate passes, stop this capacity study and record the result. If one does, retain a development candidate only. Any earlier-origin extension needs its own frozen scope; earlier exposed years are replication across conditions, not untouched confirmation. Do not headline 11,550 “new assessments” by filling earlier years with unchanged control predictions.

### 8.2 Experiment B — training-only native-expert distillation into one small student

**Priority: conditional second. Status: proposed, not run.** This tests KD, not a new sports target or automatic full-ensemble compression.

**Teacher contract**

Use the three already trained native history members from the closed TRAIN recipe, including their TRAIN-learned eta and TRAIN-only dependencies. Define the teacher explicitly as the equal mixture of their **uncalibrated legal terminal distributions**. Verify those distributions against the native terminal-likelihood mathematics before generating targets. Do not use outer-CAL temperature, dependence refits, mixtures, final C0 probabilities or the new connector to create student TRAIN targets.

This is a deliberately identified **neural-group teacher**, not full C0, not the operational hybrid and not a market teacher. Its equal mixture is only a frozen supervision source; the deployed research candidate would be a learned student, not another proposed fixed probability blend. If the training-only distribution/dependency contract cannot be verified, this study does not start.

**Matched student comparison**

Use the fixed width16 native student at12 epochs, with the same inputs, legal terminal output, auxiliary head and optimizer policy in both arms:

- Control: terminal one-hot NLL +0.1 masked auxiliary SmoothL1.
- KD: **0.5 terminal one-hot NLL +0.5 soft terminal cross-entropy +0.1 the same auxiliary loss**.

Temperature1 and λ=0.5 are declared design choices, not literature-derived optima. No temperature/λ/pseudo-data grid. Reuse the three width16/epoch12 control checkpoints from Experiment A only if the complete recipe and initialization provenance match; otherwise count their three refits explicitly. Add **three student fits** for KD, one per paired seed. No teacher refit is planned.

Prespecify seed20260910 as the single-student candidate; the other two seeds test repeatability. Do not choose the best seed after assessment or quietly ensemble three students while claiming single-student inference. A future seed-ensemble deployment would be a different cost/quality contract.

**What to measure and what would count**

The primary contrast is KD student versus the same small directly trained student. Compare both with the fixed teacher and the unchanged assembled C0. Calibrate only after student training, using the same outer-CAL policy; preserve the unchanged rating/MLP experts when evaluating a student replacement for the neural group. Report winner LL/Brier, legal scoreline NLL, calibration/support profiles, every seed and the same fixed sports guards. Teacher agreement is diagnostic, never the success criterion.

Measure end-to-end feature+prediction latency and memory on the same machine/workload, not just neural parameter counts. For an accuracy-development claim, require a negative upper monthly LL interval versus the matched small control, nonworse Brier point and the inherited guards. For a compression-only claim, require predeclared C0 noninferiority and a real measured cost reduction; no claim of improved probabilities. No financial/tournament certification follows from either.

This design asks whether the existing teacher's richer distribution is usefully compressible. It does not add real sports information, certify the full model's joint output, or promise to remove selected EV bias.

### 8.3 What not to queue automatically

| Tempting action | Why not next |
|---|---|
| Another winner/scoreline, return-MSE, focal, smoothing, CLOSE or profit-head sweep | Closely related controlled studies already exist; no newly isolated mechanism justifies repeating them. |
| Universal probability flattening | Can increase underdog value claims. Checked example: p=.20, odds5, tax12%, T=2 moves p to1/3 and EV from−12% to+46.67%. |
| More history dropout or reconstruction masks | Tested recipes failed; no new mechanism established. |
| Another H365/monthly refit grid | The native factorial already rejected that decay recipe; cadence-only gain was inconclusive. |
| A larger Transformer/foundation model by reputation | No isolated capacity diagnosis; chronological, proper-score and complete-input controls are required. |
| Fit market shrinkage or H immediately | The old archival matrix exists; the new strict G0 contract remains unfulfilled. This research does not authorize its fitting. |
| Improve bookmaker agreement to a chosen percentage | Agreement is not truth, a proper score, or incremental value. |
| Treat auxiliary gradient conflict as a cause | Conflict alone need not harm generalization.[12] |
| Generate arbitrary synthetic rosters/history columns for KD | Violates relational/temporal input semantics; external tabular CutMix evidence does not validate it. |
| Reopen the old10/10 loop | Explicitly stopped; no automatic attempt11. |

### 8.4 Keep evidence acquisition separate from sports experimentation

Plan09 already completed local exact-byte capture, dual clocks, immutable opportunity/missingness records and future-valid fixed-artifact software mechanics. Repeating those as the next implementation task would ignore completed work. Authentic source authorization, retention/quotas, official locators, participant bindings, start/result evidence and real live-input support remain blockers to activation. No external probes ran here.

The planned forward start and expiry must not be silently moved. The genuine prospective-forecast count remains zero. Continue bounded historical sports research if authorized, while keeping any later financial claim behind its actual evidence and precision requirements. More elapsed observations do not make a persistent bias disappear, and an empty/no-bet policy does not satisfy a reliability target.

## 9. Contradictions resolved and uncertainty retained

| Apparent contradiction | Resolution |
|---|---|
| Scoreline-vs-winner was inconclusive, but the report says terminal wins | Different recorded recipes: earlier comparison inconclusive; later matched native18-fit study modestly favors terminal. Preserve both. |
| H has not run, but thousands of heads are archived | The archival H/objective matrix ran. The newer strict G0-admitted H/control protocol did not. |
| A tiny network cannot overfit | Parameter count alone does not establish effective complexity, independent sample size or generalization. |
| Twelve beats six, so the model is too small | This tests duration under one architecture, not width or irreducible error. |
| KD sometimes beats its teacher, so copying bias cannot matter | Finite-sample regularization may help; fixed-teacher population identities still expose inherited bias. |
| In-sample teacher targets are always leakage | Closed-TRAIN KD is legitimate; outer-CAL/assessment contamination and false historical issuance are not. |
| Selected market probabilities predict fewer wins than occur | The market residual is uncertain; market is a comparator, not known truth. The model's much larger selected overstatement remains. |
| Better LL should improve betting | A global score and a price-conditioned selection policy answer different questions; taxes, prices and admitted timing matter. |
| Sports research must wait for a multi-year forward plan | No. Historical development and prospective financial certification are separate streams. |

What remains unknown: the causal source of intact-input forecast error; whether a matched width/duration change helps; the optimal finite-sample student; the end-to-end value of compression; authentic forward selection reliability; and official-edition tournament quality. This research supplies discriminating next tests rather than declaring those unknowns solved.

## 10. Verification, artifacts and boundaries

Newly executed checks:

- **9,690 per-row loss values**, nine group summaries, **72 paired-score interval objects** and **12 selected-error interval objects** independently recomputed.
- **10,527 scalar checks**, maximum absolute difference **3.33e−16**. Independent `math.log`/`math.fsum` losses and explicit bootstrap frequency-matrix aggregation; the same prescribed random draws are intentional. Checks are arithmetic assertions, not independent observations.
- **130 mathematical cases** for return-MSE weighting, mixed hard/soft CE, scoreline nonidentifiability, flattening and conditioning; maximum return-identity numerical error **3.55e−15**.
- Actual checkpoint tensor enumeration and a separate dimension formula agree. The source-only overcount was corrected; the failed receipt-key lookup and count assertion are retained in the capacity receipt's verification note.
- Both generated figures were opened and inspected. No synthetic validation curve or trained width16/64 result is shown.
- **28 preserved hashes:** seven frozen inputs, three native checkpoints, one shared checkpoint and seventeen archived implementation files. A fresh disk-only smoke bound all41 source columns on1,615 accepted rows to the original table, checked the frozen candidate's labels/BO/probabilities exactly, and verified all330 chosen-side probabilities/outcomes and the76-row near band.
- Citation CLI **PASS:**36 cited sources, no missing/unused entries or placeholder URLs. The access ledger distinguishes30 relevant-full-text inspections from6 abstract/publisher-summary inspections.
- Two independent reviews found no material issue in the inspected statistical/design and high-impact source claims. Their bounded scope is preserved; neither is a full raw-source or application audit.

Artifacts under `data/08_reporting/model_redesign_research/model_redesign_research_20261004_01/`:

| File | Content |
|---|---|
| `matched_diagnostic_rows.parquet` | The exact accepted1,615 rows and row losses used for these diagnostics. |
| `conditional_scores.csv` | Nine fixed groups, LL/Brier and counts. |
| `conditional_contrasts.csv` | Seventy-two paired block-bootstrap contrasts. |
| `selected_error_decomposition.csv` | Twelve algebraic component/total intervals on fixed330 and76. |
| `native_capacity_inventory.json` | Checkpoint identities, hashes, tensor shapes, counts and recorded TRAIN curves. |
| `mathematical_checks.json` | Tested identities/counterexamples and proposed width-count formula. |
| `independent_verification.json` | Numerical oracle scope and observed maximum error. |
| `source_ledger.json` | Thirty-six citations, access levels, scoped claims and short excerpts where available. |
| `conditional_diagnostics.png`, `native_training_curves.png` | Visualizations of saved observations only. |
| `reproduction_recipe.json`, `verification_notes.json` | Exact inputs/formulas, implementation hashes and resolved verifier/tool incidents. |
| `research_reviews.json`, `citation_verification.json` | Independent review scope and observed citation CLI result. |
| `verification.json` | Source-bound disk smoke, arithmetic receipts, local-link and documentation checks. |

The source bibliography is not a claim that every linked paper was fully reviewed cover to cover. The automated citation utility checks reference syntax and missing/unused IDs, not factual truth, HTTP availability, publication quality or causal transfer. Source access and high-impact claims were checked separately.

No new application tests, neural training, benchmark refits, cash-ledger replay or deployment checks are claimed. Older test/replay counts belong to their original receipts. All evaluated years remain exposed development; actual lineups and daily clocks remain scenarios, not timestamp-certified issuance. No new model was promoted and no automatic follow-up is running.

## Bibliography

[1] Gneiting and Raftery (2007). **Strictly Proper Scoring Rules, Prediction, and Estimation.** https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

[2] Gneiting, Balabdaoui and Raftery (2007). **Probabilistic forecasts, calibration and sharpness.** https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf

[3] Gneiting and Resin (2021, version3). **Regression Diagnostics meets Forecast Evaluation: Conditional Calibration, Reliability Diagrams, and Coefficient of Determination.** https://arxiv.org/pdf/2108.03210v3

[4] Giacomini and White (2006). **Tests of Conditional Predictive Ability.** https://economia.uc3m.es/jgonzalo/teaching/PhdTimeSeries/GiacominiWhite.pdf

[5] Smith and Winkler (2006). **The Optimizer’s Curse: Skepticism and Postdecision Surprise in Decision Analysis.** https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf

[6] Dwork et al. (2015). **Generalization in Adaptive Data Analysis and Holdout Reuse.** https://arxiv.org/html/1506.02629v2

[7] Király and Qian (2017). **Modelling Competitive Sports: Bradley-Terry-Élő Models for Supervised and On-Line Learning of Paired Competition Outcomes.** https://arxiv.org/pdf/1701.08055

[8] Reade, Singleton and Vaughan Williams (2020). **Betting markets for English Premier League results and scorelines: evaluating a forecasting model.** https://www.reading.ac.uk/web/files/economics/emdp202003.pdf

[9] Broatch and Karl (2017). **Multivariate Generalized Linear Mixed Models for Joint Estimation of Sporting Outcomes.** https://arxiv.org/pdf/1710.05284

[10] Tsokos et al. (2019). **Modeling outcomes of soccer matches.** https://link.springer.com/article/10.1007/s10994-018-5741-1

[11] Lopez-Paz, Bottou, Schölkopf and Vapnik (2015/2016). **Unifying distillation and privileged information.** https://arxiv.org/html/1511.03643

[12] Jiang et al. (2023). **ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning.** https://proceedings.neurips.cc/paper_files/paper/2023/file/60f9118a849e8e9a0c67e2a36ad80ebf-Paper-Conference.pdf

[13] Donti, Amos and Kolter (2017). **Task-based End-to-end Model Learning in Stochastic Optimization.** https://arxiv.org/pdf/1703.04529

[14] Elmachtoub and Grigas (2017 preprint / 2022 publication). **Smart “Predict, then Optimize”.** https://arxiv.org/pdf/1710.08005v5

[15] Hinton, Vinyals and Dean (2015). **Distilling the Knowledge in a Neural Network.** https://ar5iv.labs.arxiv.org/html/1503.02531

[16] Bucilă, Caruana and Niculescu-Mizil (2006). **Model Compression.** https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf

[17] Furlanello et al. (2018). **Born Again Neural Networks.** https://proceedings.mlr.press/v80/furlanello18a.html — Abstract/publisher-summary access only.

[18] Mirzadeh et al. (2019/2020). **Improved Knowledge Distillation via Teacher Assistant.** https://arxiv.org/pdf/1902.03393

[19] Ji and Zhu (2020). **Knowledge Distillation in Wide Neural Networks: Risk Bound, Data Efficiency and Imperfect Teacher.** https://proceedings.neurips.cc/paper_files/paper/2020/file/ef0d3930a7b6c95bd2b32ed45989c61f-Paper.pdf

[20] Guo, Pleiss, Sun and Weinberger (2017). **On Calibration of Modern Neural Networks.** https://proceedings.mlr.press/v70/guo17a.html — Abstract/publisher-summary access only.

[21] Kim, Park, Lee and Kwak (2025). **The Role of Teacher Calibration in Knowledge Distillation.** https://arxiv.org/html/2508.20224v1

[22] Jeong, Lee, Lee and Yoo (2026-10-01). **Distillation of Tabular Foundation Models into Efficient Predictors.** https://arxiv.org/html/2610.01435v1

[23] Viering and Loog (2021 preprint / 2022 review). **The Shape of Learning Curves: A Review.** https://arxiv.org/html/2103.10948

[24] Nakkiran et al. (2019/2020). **Deep Double Descent: Where Bigger Models and More Data Hurt.** https://arxiv.org/html/1912.02292

[25] Rubachev, Kartashev, Gorishniy and Babenko (2024, version4). **TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks.** https://arxiv.org/html/2406.19380v4

[26] Cai and Ye (2025). **Understanding the Limits of Deep Tabular Methods with Temporal Shift.** https://proceedings.mlr.press/v267/cai25j.html — Abstract/publisher-summary access only.

[27] Chen et al. (2023 preprint / KDD2024). **Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift.** https://arxiv.org/html/2310.14838

[28] Gardner, Popovic and Schmidt (2023). **Benchmarking Distribution Shift in Tabular Data with TableShift.** https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html — Abstract/publisher-summary access only.

[29] Shwartz-Ziv and Armon (2021). **Tabular Data: Deep Learning is Not All You Need.** https://arxiv.org/html/2106.03253

[30] Krese and Štrumbelj (2021). **A Bayesian approach to time-varying latent strengths in pairwise comparisons.** https://pmc.ncbi.nlm.nih.gov/articles/PMC8136743/

[31] Oliveira and Newell (2024). **A hierarchical approach for evaluating athlete performance with an application in elite basketball.** https://pmc.ncbi.nlm.nih.gov/articles/PMC10799012/

[32] Scholtes and Karakuş (2024). **Bayes-xG: player and position correction on expected goals (xG) using Bayesian hierarchical approach.** https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1348983/full

[33] Do et al. (2021). **Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience in League of Legends.** https://ar5iv.labs.arxiv.org/html/2108.02799

[34] Zhao, Kim, Sahoo, Ma and Ermon (2021). **Calibrating Predictions to Decisions: A Novel Approach to Multi-Class Calibration.** https://proceedings.neurips.cc/paper_files/paper/2021/hash/bbc92a647199b832ec90d7cf57074e9e-Abstract.html — Abstract/publisher-summary access only.

[35] Rubachev, Kotelnikov, Kartashev and Babenko (2025). **On Finetuning Tabular Foundation Models.** https://arxiv.org/html/2506.08982v2

[36] Landsgesell, Knoll and Wenzel (2026-03-31). **ScoringBench: A Benchmark for Evaluating Tabular Foundation Models with Proper Scoring Rules.** https://arxiv.org/abs/2603.29928 — Abstract/publisher-summary access only.
