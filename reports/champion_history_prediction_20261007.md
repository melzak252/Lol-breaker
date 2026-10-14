# Champion-aware historical outcomes and statistics

Run: `champion_history_20261007_01`. Protocol frozen 2026-10-07; interrupted execution resumed with explicit authorization on 2026-10-08. This is an odds-free historical research challenger, not a production or betting replacement.

**COMPLETED / NOT_ADOPTED for winner, length, kills, duration and the complete bundle.** The fixed representation does not establish incremental champion value. Preserve C0 and the existing standalone prop models; do not resume this study as an unexecuted proposal.

## 1. Question and boundary

Does a shared player-history representation that learns champion–performance–pace interactions improve winner, coherent terminal score/length, kills or duration predictions? Champions are **past observations**, not the upcoming composition. Another history encoder is not automatically additional information over C0.

Keep the completed champion-pool, player/meta and stable180/meta30 negatives, original standalone-prop study and actual-composition diagnostic separate. This study trains directly on the requested targets; it does not apply another scalar correction to C0. C0 probabilities are only a frozen comparison, not model inputs, map-win probabilities or new training targets.

Known first-map rosters and strict prior-day releases are conditional retrospective scenarios. They are not authenticated roster announcements, exact publication clocks or issued forecasts. Historical 2026 and every earlier year remain development-exposed. No target picks, bans, actual role assignments, side, pick order, performance, outcome, terminal score, length, uncertified current patch or odds enters the pre-series representation.

## 2. Source audit and target populations

Immutable GOLGG recovery: **41,915 series / 69,370 maps / 693,700 player-map observations**. Canonical first-map population: **40,636 series**, with **66,812 recorded maps**. The additional **1,279 BO2 series / 2,558 maps** remain usable historical evidence; they are not legal BO1/3/5 terminal-score targets. Whole-series releases, including delayed series, govern history membership and partition eligibility.

|Partition|Series|Recorded maps|Winner/score labels|Kills/duration labels|
|---|---:|---:|---:|---:|
|TRAIN, played and released before 2025-07-01|36,205|56,888|36,205 each|56,888 each|
|CAL, 2025H2 and released before 2026|1,328|2,960|1,328 each|2,960 each|
|Assessment, frozen canonical 2026|3,103|6,964|3,103 each|6,964 each|

No canonical target is removed for an incomplete upcoming composition. Identity, legal score and map-order validation retain explicit target-specific reasons; the measured retained population has no winner, score, kill or duration label absences. Assessment runs 2026-01-08 through 2026-09-08: BO1 **745**, BO3 **1,873**, BO5 **485**. Length scoring excludes deterministic BO1, leaving **2,358** series. Later-map statistics have **3,861** actually played maps beyond Map 1; unplayed maps do not receive fabricated labels.

Four raw champion cells are missing. On thirteen maps, thirty repeated champion-identity cells are ambiguous and quarantined **only in the champion channel**. Independently usable player statistics, map outcomes and series targets remain. Missing/unreliable champions, observed out-of-TRAIN-vocabulary champions and recognized champions retain distinct codes; unsupported and unknown-player history states remain explicit.

The rule matrix binds competition/edition/phase rather than year. It provides eleven hard-Fearless bindings among 1,257 contexts. Assessment has **185 HARD_FEARLESS** and **2,918 UNKNOWN** series; **zero evidenced STANDARD or SOFT_FEARLESS** targets. UNKNOWN must not be called standard. This source cannot establish a standard-versus-Fearless effect. No draft simulator, depletion trajectory or future draft policy is fitted.

Prepared source, schema, exclusions and provenance: [audit](../data/04_feature/champion_history/champion_history_20261007_01/audit.json), [prepared counts](../data/04_feature/champion_history/champion_history_20261007_01/prepared.json), [protocol](../data/05_model_input/champion_history/champion_history_20261007_01/protocol.json), [input/code binding](../data/05_model_input/champion_history/champion_history_20261007_01/binding.json).

## 3. Fixed representation and distribution contract

Each current player receives their last **16 strictly released past maps**, with no history-window sweep. All maps in the target series use the same pre-series representation. Historical observations contain champion and role, twelve performance channels with masks, own/opponent/total kills, duration in minutes, pre-game opponent strength, observation/release age and historical patch/competition. Current target patch is descriptive metadata only. Vocabulary and masked normalization use closed TRAIN only.

A shared width32 temporal encoder uses champion embeddings interacting with performance, mask-aware historical attention and past-role-aware roster aggregation. Individual player strength/support/inactivity and previously observed teammate overlap are retained without organization identity or required organization history. Strength and style are task-oriented learned representations, not identified causal factors. Transfers, new organizations and sparse/unknown player histories remain supported input states.

Every recipe has the same nominal **41,100 parameters**. Effective active capacity differs: champion information is zeroed in the hidden-champion controls; score-only training does not activate the prop objective/head. Equal nominal parameter count is not an equal-active-capacity experiment.

The terminal head normalizes only legal score bins: A wins with zero/one/two opponent wins, then the reversed outcomes. BO1 supports bins0/3, BO3 bins0/1/3/4 and BO5 all six. Winner, length and each map's existence probability are derived from this single PMF. There is no second BO projection.

The multitask heads predict five map-index **negative-binomial kill** and **lognormal duration** distributions. Saved outputs include only legal potential indices for each BO. Their forecasts are conditional on the map being played; `probability_played` is exported separately. Shared encoding does not certify a joint kill–duration, outcome–statistics or parlay law.

Three seeds contribute complete distributions with equal weights. Score calibration tempers each seed's legal PMF before mixing; kill/duration calibration scales dispersion, not means. NB/lognormal parameters are not averaged into a substitute parametric distribution.

## 4. Frozen factorial, budgets and interruption

|Variant|Historical champion channel|Training objectives|
|---|---|---|
|A|Hidden|Terminal score|
|B|Included|Terminal score|
|C|Hidden|Terminal score + kills + duration|
|D|Included|Terminal score + kills + duration|

Three paired seeds:20261007/20261008/20261009; one fixed twelve-epoch duration, AdamW, batch256, learning rate0.001, weight decay0.0001, prop weights0.25 each. No width/window/epoch/feature-family retry or assessment fit.

A separate actual TRAIN-only software smoke used256 series/656 maps, one epoch/four updates and saved-model replay with maximum error0. Repeating that saved object as three components exercised the ensemble interface, not three neural fits.

A host restart interrupted the original first-seed D process after A/B/C were saved. No D epoch/optimizer checkpoint survived. The user explicitly authorized **“Resume the identical recipe”**. Original execution bytes and the three original model bytes are retained; completed checkpoints are restored without retraining. The lost D attempt is repeated with the same seed, closed data, architecture, objectives, optimizer and duration. Authorized limits distinguish **12 completed trajectories / 144 completed seed-epochs** from **13 formal fit attempts**, including **one interrupted partial attempt with unknown completed progress**. It is not twelve attempts or a thirteenth adaptive recipe.

CAL permits four score temperatures plus kill and duration dispersion scales for C/D: **eight ensemble-level scalar fits**. Bounds[0.25,4], quadratic log-scalar penalty0.001. CAL selects B versus D separately for winner and length; D is the fixed champion-aware statistical candidate. C0 and assessment fits are zero.

[Original interrupted journal](../data/08_reporting/champion_history/champion_history_20261007_01/execution_interrupted.json), [explicit recovery authorization](../data/05_model_input/champion_history/champion_history_20261007_01/resume_authorization.json), [retained incidents](../data/08_reporting/champion_history/champion_history_20261007_01/orchestration_incidents.json).

## 5. References, proper scores and frozen advancement gates

- Winner: unchanged native C0 W32/E12/full on the same3,103 targets.
- Length: original selected three-seed width128/epoch12 standalone model on2,358 BO3/5 targets.
- Map1 kills: original ridge1 negative-binomial model.
- Map1 duration: original15-leaf LightGBM/lognormal model.
- Coherent-score diagnostic: existing fixed Markov beta0.35 conditional law preserving the frozen C0 series-win margin. It is a heuristic reference, not a newly trained map-outcome model or certified joint truth.
- Matched champion contrasts:B−A and D−C. Multitask contrasts:C−A and D−B are separate diagnostics.

Winner scores use LL/Brier, discrimination, calibration bins and extreme errors. Terminal score uses categorical NLL; length uses NLL/RPS. Statistics use NLL/CRPS, predictive-mean MAE and central50%/80% coverage/width. All-map primary statistics average available played-map scores **within each series**, then weight series equally; map-weighted summaries are separate.

The twelve primary NLL contrasts were frozen before fitting. Each uses5,000 paired **whole-series monthly-block** draws, all nine monthly universes, match-weighted and equal-month estimands. Two-sided Bonferroni confidence is **99.583333%**, not adjustment for all adaptive historical research. Both required NLL upper bounds must be strictly below zero. Brier/RPS/CRPS must not increase pointwise against both corresponding controls. Coverage guards require at least45%/75% at nominal50%/80%, and no more than2.5pp worsening of absolute nominal-coverage error against either control. These are declared research guards, not population guarantees.

Existing prop references cover Map1 only. First-map existing-reference comparisons and all-map matched-control comparisons are distinct populations, never silently intersected or presented as existing later-map benchmarking. Later-map legacy-reference absences are explicit. A task advances independently of other tasks; complete-bundle advancement additionally requires all four task gates and one common coherent winner/length variant. No post-assessment reselection or outcome-selected league exclusion is allowed.

## 6. Completed fitting, calibration and selection

All twelve trajectories completed twelve epochs; three original checkpoints were restored and nine fits completed after the explicitly authorized recovery. Accounting is **13 actual formal attempts / 12 completed trajectories / 144 completed seed-epochs**, including one interrupted partial attempt of unknown progress. Eight ensemble CAL scalars completed; zero C0, assessment or scoring-repair fits. The separate software smoke remains one TRAIN-only neural fit.

|Recipe|Score temperature|NB dispersion scale|Lognormal sigma scale|
|---|---:|---:|---:|
|A|1.015573065|—|—|
|B|1.063410602|—|—|
|C|0.968728613|0.716265947|0.933284449|
|D|0.997346213|0.709311445|0.939455286|

CAL alone selected **D for winner, B for length, D for kills/duration**. CAL winner NLL B/D is0.551603094/0.548424283; length NLL B/D is0.792437376/0.793916007. These task choices do not create a mixed D-winner/B-length score law. Every saved ensemble remains internally coherent; no mixed bundle is adopted or served.

## 7. Measured assessment results

### 7.1 Winner, terminal score, length and map existence

|Same assessment series|Winner LL|Brier|AUC|Terminal-score NLL|Length NLL, BO3/5|Length RPS|
|---|---:|---:|---:|---:|---:|---:|
|A: no champions, score-only|0.546850131|0.185004453|0.794721|1.086631318|0.718416382|0.263303080|
|B: champions, score-only|0.550086055|0.186083727|0.792402|1.088256422|0.716463051|0.262291162|
|C: no champions, multitask|0.546201388|0.184960052|0.794408|1.084146553|0.716365495|0.262381429|
|D: champions, multitask|0.546320524|0.184929253|0.794514|1.083563982|0.715246349|0.261724544|
|Unchanged C0 + diagnostic coherent law|0.535448701|0.181034493|0.802888|1.065382436|0.703169303|0.255534705|
|Existing standalone props|Unsupported|Unsupported|Unsupported|Unsupported|0.714579615|0.261305154|

Winner/score use3,103 rows; length uses2,358. C0's score and length columns come from the existing beta0.35 conditional reference law, not a new native score predictor. The reference never supplies map-win inputs to the challenger.

Descriptive ECE10 in this study's raw A/B orientation is A/B/C/D/C0 **3.3132%/2.1897%/2.8125%/3.2097%/2.4254%**. Rowwise orientation alignment changes finite-sample binary bins; these values are not a replacement for earlier native-orientation diagnostics. Proper C0 winner LL/Brier reproduce the earlier reference. Accuracy A/B/C/D/C0 is72.2204%/71.6081%/71.4792%/71.3503%/72.0593%. Winner LL≥2.5 counts are13/15/13/15/12; maximum winner LL3.777354/3.493203/3.628603/3.767308/4.349928. All winner/score losses are finite. These diagnostics neither override LL/Brier nor identify a cause.

Map existence is scored on all **8,789 legal series/map-index pairs**, including unplayed potential maps. D's pair-weighted Brier is0.070218054 and equal-series Brier0.057268836; its Map3/4/5 Briers are0.172633280/0.247535115/0.185613752. Map1, and Map2 in BO3/5, exist deterministically. Conditional statistic scores use only played maps, not this potential-map population.

### 7.2 First-map distributions: identical3,103 targets

CRPS, MAE and widths use kills or minutes, respectively. NLL is the exact mixture likelihood.

|Task / recipe|NLL|CRPS|Mean MAE|50% coverage / width|80% coverage / width|
|---|---:|---:|---:|---|---|
|Kills, existing ridge/NB|3.590488045|5.010184747|7.048286|50.0806% /10.9981|79.5682% /20.9346|
|Kills, C|3.596492400|5.031458699|7.039481|51.9497% /11.0944|79.6004% /21.1163|
|Kills, D|3.594386287|5.027001869|7.047152|51.1118% /11.0812|79.4393% /21.1170|
|Duration, existing LightGBM/lognormal|3.076194163|3.011501640|4.237251|46.5034% /6.5349|75.9910% /12.4755|
|Duration, C|3.065195468|2.996669968|4.259925|46.8901% /6.6638|76.6677% /12.7464|
|Duration, D|3.065795804|2.998932615|4.268903|46.8579% /6.6647|76.8933% /12.7452|

D improves duration NLL/CRPS against the existing model, but C without champion information is better. D's predictive-mean duration MAE is also worse than the existing reference. This is not evidence that champions supplied the improvement.

### 7.3 All played maps: matched C/D comparison

Primary all-map metrics average within series first: **6,964 observed maps /3,103 series**. Existing prop models have no later-map statistical reference.

|Task / recipe|Series-weighted NLL|CRPS|Mean MAE|50% coverage / width|80% coverage / width|
|---|---:|---:|---:|---|---|
|Kills, C|3.609365235|5.092503425|7.128855|52.5540% /11.3259|80.3395% /21.5592|
|Kills, D|3.608718568|5.093432708|7.143758|51.9046% /11.3168|80.0510% /21.5588|
|Duration, C|3.060593413|2.980976546|4.237881|47.6969% /6.7598|77.0453% /12.9330|
|Duration, D|3.062873762|2.988285324|4.253899|47.4213% /6.7452|77.1554% /12.9037|

Separate map-weighted C/D kill NLL is3.609496836/3.609003502 and duration NLL3.056580783/3.058114336. Weighting longer series more heavily does not change the frozen primary estimand.

### 7.4 All twelve primary contrasts

Negative is better. Every interval is99.583333% with5,000 paired whole-series monthly draws and nine blocks; length excludes745 deterministic BO1 targets. All paired eligible populations have zero candidate/control prediction absences.

|Candidate−control / metric|Match-weighted Δ [interval]|Equal-month Δ [interval]|
|---|---|---|
|B−A winner|+0.003235924 [−0.001371670,+0.007476341]|+0.003284268 [−0.000773189,+0.007624347]|
|D−C winner|+0.000119136 [−0.002581560,+0.002898831]|−0.000662092 [−0.005339026,+0.003174403]|
|B−C0 winner|+0.014637354 [+0.001549746,+0.030385994]|+0.015404467 [+0.003853260,+0.028658436]|
|D−C0 winner|+0.010871824 [+0.003984947,+0.019233993]|+0.010301080 [+0.004028068,+0.018865654]|
|B−A length|−0.001953331 [−0.003270407,−0.000267340]|−0.001642231 [−0.003100264,−0.000073225]|
|D−C length|−0.001119147 [−0.002201738,+0.000647699]|−0.001123411 [−0.002814499,+0.000571423]|
|B−old length|+0.001883436 [−0.004641575,+0.010777129]|+0.003752752 [−0.004212097,+0.011828303]|
|D−old length|+0.000666734 [−0.007496680,+0.009727463]|+0.002423372 [−0.005175266,+0.011537239]|
|D−C kills, all maps|−0.000646667 [−0.010294377,+0.007083927]|−0.000102005 [−0.010451694,+0.006653058]|
|D−old kills, Map1|+0.003898242 [−0.013513422,+0.020172447]|+0.002106986 [−0.017764656,+0.019140164]|
|D−C duration, all maps|+0.002280349 [−0.002383326,+0.006798276]|+0.001296974 [−0.003542376,+0.005811853]|
|D−old duration, Map1|−0.010398360 [−0.023691321,−0.004602403]|−0.015022996 [−0.032452695,−0.004885804]|

B has an incremental length signal over A, but does not beat the existing length model. D's first-map duration signal over the existing model does not beat C. The candidate winner models are worse than C0 on this development population; no universal architecture or champion-history rejection follows.

Multitask winner diagnostics use separate95% intervals: C−A **−0.000648742 [−0.005267581,+0.003030721]**, D−B **−0.003765530 [−0.009923879,+0.001110033]**. Their match-weighted intervals cross zero. D−B equal-month winner interval is negative, but this separate estimand does not override the primary comparison family. No demonstrated winner improvement from multitask supervision is established.

## 8. Task and complete-bundle decisions

|Task|CAL-selected candidate|Research decision|Failed requirements|
|---|---|---|---|
|Winner|D|**NOT_ADOPTED**|Both required improvement intervals fail; Brier rises versus C0|
|Series length|B|**NOT_ADOPTED**|Existing-reference NLL interval crosses zero; RPS rises versus existing model|
|Kills|D|**NOT_ADOPTED**|Both improvement intervals cross zero; CRPS increases against both references|
|Duration|D|**NOT_ADOPTED**|Matched-control NLL interval crosses zero and point delta is positive; CRPS rises versus C|
|Complete coherent bundle|None|**NOT_ADOPTED**|All four task gates fail; selected winner/length recipes also differ|

Declared coverage guards pass for the two statistical tasks, but this does not compensate for failed accuracy guards or certify nominal population coverage. Duration80% coverage remains below80%. Decisions are completed negative research outcomes, not an unfinished evidence collection queue or proof that every alternative representation is useless.

## 9. Profiles and absence boundaries

Recorded map-index counts1–5 are **3,103 /2,358 /1,122 /262 /119**. D kill NLL by index is3.594386/3.613840/3.642894/3.653375/3.477089; duration NLL3.065796/3.041153/3.062570/3.107150/3.043939. These are conditional-on-played, map-weighted descriptive slices, not scores on unplayed maps.

Hard-Fearless covers185series/493maps; UNKNOWN covers2,918series/6,471maps. There is no evidenced standard control. On the hard-Fearless slice, C/D equal-series duration NLL is3.039789/3.027591, but winner NLL0.574374/0.579258. Neither this small descriptive slice nor the current rule evidence establishes a Fearless-specific benefit. No rule or league exception is fitted.

All BO, map-index, monthly, descriptive target-patch, historical support, roster and missingness profiles remain in `profiles.csv`. Supplemental profiles add exact supplied competition context, past inactivity, valid-token twelve-stat coverage, unfamiliar-player support and champion-pool breadth. **6,360 supplemental task/scope/group rows** are descriptive, not6,360 independent experiments.

Supplemental definitions were recorded separately during resumption. Their original `frozen_at` refers to the initial pre-pool definition; pool/absence amendments retain earlier source/binding hashes, but their exact amendment clock was not separately retained. They are descriptive only and are **not retroactively certified as pre-fit or pre-assessment**; see [timing note](../data/08_reporting/champion_history/champion_history_20261007_01/profile_binding_timing_note.json). The original primary contrasts and gates remain frozen.

Assessment support is full/partial/cold **1,858/1,241/4** series; new/established roster746/2,357. At least one current player lacks released history on334series. Maximum supported-player inactivity bins0–7/8–30/31–90/91+ have1,770/570/155/274series, with330partially unsupported and4fully unsupported kept separate. Valid-token stat coverage is≥90% on2,867series,50–90% on232 and unavailable on4cold rows. Padding never counts as missing observed statistics.

Minimum resolved champion breadth across ten last16 histories is1–3/4–6/7+ on415/1,164/1,180series;334have an unsupported player and10lack a resolved champion for at least one supported player. This is a **lower bound**, not full career breadth: out-of-TRAIN identities share UNK and are excluded from distinct-known counts.

The per-target absence ledger contains55,712 **head/task absence rows**, not55,712 missing fixtures: A/B/C0 lack statistical heads, legacy props lack later-map statistics and winner/terminal-score heads. C/D have zero label/prediction absences on every requested eligible assessment task. BO1 length exclusions are deterministic targets, not unavailable predictions.

## 10. Verification, retained failures and precision limits

**161 focused/new-plus-existing tests passed**, ten existing warnings (Starlette deprecation, LightGBM feature-name warnings and the old prop-model read-only tensor warning). This is not the full application or database suite. Real prepare, freeze, TRAIN-only smoke, formal training, score continuation, central benchmark, profile export and saved inference interfaces ran; no simulator was required or invoked.

The central `run_model_benchmark.py --champion-history` mode recomputed all3,103 assessment series,6,964 maps and twelve5,000-draw contrasts in56.15seconds, with1,176 numeric replay comparisons and maximum error0. This uses the shared model/evaluator; it is not the independent oracle.

The independent numerical oracle replayed all twelve saved networks on the complete CAL and assessment populations, reconstructed TRAIN-only preprocessing, paired initializations/shuffles, optimizer receipts, complete seed-mixture distributions, all eight CAL scalar optima, scores/CRPS/quantiles/RPS, profiles, primary/equal-month intervals, selection and every task/bundle gate. **4,474 checks /3,199,609 compared values**; integer/preprocessing maxerror0, CAL scalar2.72556e−8, objective2.53975e−11, prediction3.33067e−16, metric3.90799e−14, bootstrap3.57353e−16. Candidate forward, predict, preprocessing and evaluator methods were not reused. Torch/NumPy/SciPy primitives and bounded optimization are shared and disclosed; old props/C0 reference inputs were not independently refitted. Counts include repeated comparisons, not independent sports examples.

The temporal oracle rebuilt full-source historical membership, identities, roles, masks, vocabulary, champion quarantine and releases: **51 checks,40,636 targets,66,812 maps,406,360 player slots and5,882,697 historical tokens**, including58,945 BO2-history tokens; numerical alignment error0. Another **31 checks** physically reversed4,947 source series, changed future records and independently mutated forbidden target fields on six targets; all pre-match inputs remained bitwise unchanged. Source iteration/container parsing and the established Gaussian strength inputs are disclosed reused infrastructure, not independent native-C0 fitting.

Saved typed inference exercised twelve seed models and four ensembles: **2,656 checks** across actual cold/new/partial-support cases, generated fully unknown rosters, unknown/missing champions and statistics, swaps, player-slot permutations, legal mass and winner/length/play coherence, disabled-head rejection and input/model nonmutation. TRAIN preprocessing maxerror1.74909e−11. Largest raw-parameter swap/permutation absolute error7.6293945e−6 is within the declared2e−4 parameter check; probability checks use2e−6. These are bounded FP32 checks, not exact invariance of every tensor or every matchup.

Parent public verification ran nine actual B/C/D CLI predictions across BO1/3/5 and three expected rejections (pre2026 origin, incompatible benchmark mode and altered bootstrap count): **279 PMF/derived/CDF comparisons, maxerror0** against same-batch saved inference.

**A stricter independent public check failed and remains failed.** Four actual A/B/C/D CLI processes exited0, but six C-versus-typed checks exceeded the original1e−12 precision criterion: winner max4.5762861e−7, terminal PMF2.6861997e−7, conditional CDF3.8530420e−7. A/B/D matched. The separate pre-existing2e−6 public contract passes36checks/346values; the strict criterion was not loosened or relabeled PASS. Fresh typed repeats, compact/Fortran layouts and functional FP32 probes were exact; promoting the exact FP32 weights/normalized inputs to FP64 differed by≤1.8953821e−7. Two later fresh C CLI processes matched the typed reference but differed from the original retained output. The original bounded variation remains **causally unexplained**. Do not claim universal bitwise or1e−12 cross-process reproducibility.

The original post-fit scorer rejected C0's different participant orientation. Canonical participant IDs prove **1,524 direct /1,579 reversed /0 unmatched** rows; labels validate lineage only after ID-derived alignment and never choose probability sign. Original runner bytes and partial outputs were archived. Two scoring-only continuation failures were retained: empty-component singleton export (41.62seconds), then an extracted helper's missing `run_id` scope (44.11seconds). Successful unchanged-data/model/CAL/gate scoring took39.02seconds. These repairs added **zero fits**; AST checks prove the original fit/CAL and metric/contrast/gate statements unchanged. The explicit continuation binding preserves the original frozen protocol and binding rather than silently changing their hashes.

Other retained verifier incidents: null NaN-versus-None comparison on42cold metadata rows; a future fixture selected by nonchronological canonical row order instead of actual latest date; a static verifier-only age/tanh parenthesis error corrected before numerical execution. These are not neural recipe retries or new assessment fitting. Original failed receipts and executed/reconstructed verifier source provenance remain disclosed.

The earliest temporal receipt's end-of-run code-hash drift also remains disclosed. Original source was reconstructed by reversing recorded owned edits and archived separately; later runs archive source before execution. Hash correspondence is bounded provenance, not cryptographic execution-continuity certification.

## 11. Representation, source and market limitations

Closed-TRAIN vocabulary has no valid UNK1 tokens for champion, historical patch or exact competition embeddings. UNK1 receives CAL/assessment counts champion1,642/13,662, patch104,601/418,999 and competition116,541/422,104. Enabled UNK embedding rows and champion-UNK token-status weights have zero final AdamW first/second moments; their saved values reflect initialization/weight decay, not informative TRAIN learning. Missing and unseen identities remain distinct input states, but this design has **untrained OOV representation channels**, especially for later patches/editions. This is a substantive limitation, **not an established cause** of the negative results and not authorization to retrofit features or run another recipe.

The independent raw audit found55 player-kill-sum versus recorded team-kill mismatches among135,390 complete comparable five-player sides. Existing labels remain the sum of recorded team kills; no outcome-selected correction, deletion or changed target definition was applied. Four missing and thirty unreliable historical champion cells are masked only in that channel. Normalized raw games have no ban sequence fields; forbidden-ban perturbations use isolated synthetic sentinels. No complete draft-order or ban-sequence model is claimed.

The previous historical market result is preserved: standalone series-length binary-line LL0.669133345 versus Pinnacle0.658158740 on1,318 fixtures; Δ+0.010974605 with97.5% interval[+0.003025793,+0.023563745],423 missing actual-start clocks. Two October kill-price examples fall beyond this September8 frozen sports source; duration prices were absent. This study performs **no new bookmaker/ROI comparison**, source activation or financial certification. Missing suitable prices do not prevent sports evaluation, but do prevent bookmaker-superiority and betting-readiness claims.

No genuinely future forecasts, authenticated pre-match roster issuance, live feature adapter, live scraping, paid purchases, server/database mutations, deployment or production promotion. Unknown format is not standard. No joint kill–duration or parlay calibration. Keep all prior negative studies, canonical A0/C0, manifest/serving, G0, old10-of-10 and forward dates unchanged. This completed fixed experiment gives no automatic follow-up sweep.

## 12. Reproducible artifacts and interfaces

- Inputs/audit: `data/04_feature/champion_history/champion_history_20261007_01/{series.parquet,maps.parquet,arrays.npz,audit.json,prepared.json}`.
- Frozen protocol/original binding and recovery/scoring lineage: `data/05_model_input/champion_history/champion_history_20261007_01/`. Original protocol SHA256 `21bf3a9d8672720c23eb503fe154f5201e7cca9aa2b813edcd2c36aa767a21df`.
- Twelve versioned seed models, four complete-distribution ensembles and CAL selection: `data/06_models/champion_history/champion_history_20261007_01/`. Retain the negative models for reproducibility, not serving.
- Complete legal-score, potential-map component distributions, played-map/series scores and explicit head absences: `data/07_model_output/champion_history/champion_history_20261007_01/`.
- [Execution](../data/08_reporting/champion_history/champion_history_20261007_01/execution.json), [metrics](../data/08_reporting/champion_history/champion_history_20261007_01/metrics.json), [primary contrasts](../data/08_reporting/champion_history/champion_history_20261007_01/comparisons.json), [task/bundle decision](../data/08_reporting/champion_history/champion_history_20261007_01/decision.json), [profiles](../data/08_reporting/champion_history/champion_history_20261007_01/profiles.csv), [supplemental profiles](../data/08_reporting/champion_history/champion_history_20261007_01/supplemental_profiles.csv), [profile receipt](../data/08_reporting/champion_history/champion_history_20261007_01/profile_receipt.json).
- [Independent numerical receipt](../data/08_reporting/champion_history/champion_history_20261007_01/verification/numerical_receipt.json), [full source receipt](../data/08_reporting/champion_history/champion_history_20261007_01/verification/temporal_data.json), [future/target perturbations](../data/08_reporting/champion_history/champion_history_20261007_01/verification/temporal_perturb.json), [typed invariants](../data/08_reporting/champion_history/champion_history_20261007_01/verification/temporal_saved-typed.json), [retained strict public failure](../data/08_reporting/champion_history/champion_history_20261007_01/verification/temporal_saved.json), [bounded public contract](../data/08_reporting/champion_history/champion_history_20261007_01/verification/temporal_public_precision_contract.json).

Recorded prepare/freeze/smoke/run/recovery/score commands are in [scripts/README.md](../scripts/README.md#champion-aware-historical-outcomes-and-statistics--2026-10-07). Existing namespaces and receipts must not be reused. Safe saved inference needs the frozen source/bindings and research-root resolver:

```bash
.venv/bin/python scripts/train_champion_history.py predict --run-id champion_history_20261007_01 --variant D --match-id 80922 82251 82590
# A new nonexistent output directory is required for an authorized rescore.
.venv/bin/python scripts/run_model_benchmark.py --champion-history champion_history_20261007_01 --output-dir data/08_reporting/champion_history/champion_history_20261007_01/NEW_RESCORE
```

The central benchmark, public CLI, verification and hash closures establish bounded historical software/numerical evidence only. The scientific decision is **NOT_ADOPTED**, not a production-readiness certificate.

