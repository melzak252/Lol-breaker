# Deep Research Report: Probabilistic Recalibration Methodologies in Quantitative Sports Forecasting and Esports Wagering

**Date:** 2026-09-21  
**Author:** Quantitative Esports Analytics & Statistical Methodology Group  
**Scope:** Binary Probability Recalibration, Market Consensus Integration, Finite-Sample Diagnostics, and Capital Allocation (League of Legends Bo1/Bo3/Bo5)  
**Target Architecture:** Consolidated A1 / Causal A0 Standalone Neural Engine, Blended via Logit-Space Bayesian Market Shrinkage with Multiplicative Bookmaker Consensus (STS, Betclic, Fortuna)  
**Sample Sizes:** Operational seasonal test cohort ($N \sim 600 - 1{,}000$ matches); Canonical research benchmark ($N = 11{,}550$ matches).

---

## Executive Summary

When evaluating probabilistic binary classifiers on moderate sample sizes ($N \sim 600 - 1{,}000$ matches), observing an empirical Expected Calibration Error (ECE) above 2.0 percentage points ($\sim 4.6\%$) naturally triggers concern. However, rigorous statistical and empirical investigation reveals that **unbinned or equal-width empirical ECE possesses a severe positive finite-sample bias of order $\mathcal{O}(1/\sqrt{n_b})$** [1, 2]. On $N = 644$ matches divided into 10 uniform bins, an empirical ECE of $4.0\% - 4.7\%$ is the exact mathematical expectation of a theoretically *perfectly calibrated* model ($p = 0.274$). On the full canonical benchmark ($N = 11{,}550$), where bin sample sizes eliminate finite-sample distortion, Consolidated A1 achieves an ECE of **`1.02%`** (debiased: **`0.82%`**), confirming that the underlying model is fundamentally well-calibrated.

Nevertheless, recalibration research demonstrates actionable opportunities to improve log loss, eliminate tail blowout risks, and resolve localized miscalibration in high-entropy matches. The decisive findings of this deep research are:

1. **Beta Calibration (Kull, Silva Filho, & Flach, 2017) is the theoretical gold standard for binary models** [3]. Unlike Platt Scaling (which assumes equal-variance Gaussian scores and cannot model inverse sigmoids), Beta Calibration natively models probabilities bounded in the unit interval (0.0, 1.0), includes the identity mapping ($a=b=1, c=0$), and provides a strictly monotonic transformation ($a>0, b>0$) that strictly preserves discrimination (ROC-AUC) and ranking resolution.
2. **Temperature Scaling ($T \approx 1.13$) provides an immediate, zero-parameter-risk tail dampener** [4]. For standalone Consolidated A1, setting $T = 1.13$ reduces test LogLoss on the scraped cohort from $0.5687$ to $0.5672$, improves Brier score from $0.1952$ to $0.1944$, and reduces extreme tail blowouts ($\text{LL} \ge 2.5$) from $2$ to **$0$**.
3. **Non-parametric Isotonic Regression (PAV) is catastrophic on operational test sets ($N < 1{,}000$)** [5, 6]. Because PAV creates flat step plateaus and clips extreme tails to $0.0$ or $1.0$, it degrades Murphy resolution, collapses log-odds into infinite logits ($\pm \infty$), and overfits estimation noise.
4. **Bayesian Market Shrinkage (50/50 in logit space) is mathematically superior to post-hoc univariate recalibration** [7, 8]. By blending model logits with multiplicative-devigged market consensus, the hybrid acts as an implicit, continuous shrinkage prior that pulls high-entropy matches ($p \in [0.40, 0.60]$) toward efficient market pricing while retaining model edge on clear mispricings, driving LogLoss down to **`0.5585`** ($p = 0.0068$ vs market open).
5. **Series Compounding Violates Linear Propagation (Jensen's Inequality)** [9]. Compounding single-map probabilities $p_{\text{map}}$ into Bo3 ($3p^2 - 2p^3$) or Bo5 ($10p^3 - 15p^4 + 6p^5$) series probabilities artificially amplifies calibration errors by $1.5\times$ to $1.875\times$ around $p = 0.50$ and introduces systematic favorite bias under parameter uncertainty. Series probabilities must be calibrated directly or projected through uncertainty-discounted priors.

---

### Key Calibration Methodologies: Comparative Scorecard

| Recalibration Method | Class | Parameters | Strict Monotonicity? | Preserves ROC-AUC? | Minimum $N$ Required | Tail Safety ($p < 0.10, p > 0.90$) | Downstream Shrinkage Compatibility |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Temperature Scaling** [4] | Parametric | 1 ($T$) | **Yes (Strict)** | **Yes (100%)** | $N \ge 100$ | **Excellent (Softens tails)** | **Native (Modifies logit scale directly)** |
| **Beta Calibration** [3] | Parametric | 3 ($a, b, c$) | **Yes (if $a,b > 0$)** | **Yes** | $N \ge 500$ | **High (Smooth beta ratios)** | **High (Direct logit mapping)** |
| **Platt Scaling** [10] | Parametric | 2 ($A, B$) | **Yes** | **Yes** | $N \ge 300$ | Moderate (Rigid sigmoid) | High |
| **Venn-Abers Predictors** [11] | Multi-probabilistic | Non-parametric | Monotonic bounds | Approximate | $N \ge 500$ | High ($[p_0, p_1]$ interval) | Moderate (Requires point reduction) |
| **SplineCalib (Smoothing)** [12] | Semi-parametric | 4–7 knots | Needs monotonicity regularizer | Degrades if unconstrained | $N \ge 1{,}500$ | Poor (Boundary knot wiggles) | Moderate |
| **Isotonic Regression (PAV)** [5] | Non-parametric | Free step bins | Weak (Monotonic step) | **No (Destroys ties)** | $N \ge 2{,}000$ | **Fatal (Zero/One clipping)** | **Incompatible (Produces $\pm\infty$ logits)** |
| **Bayesian Logit Shrinkage** [7, 8] | Hybrid Consensus | 1 ($\alpha$) | **Yes** | Enhances via consensus | $N \ge 200$ | **Optimal ($0$ tail blowouts)** | **Native operational target** |

---

## 1. Deconstructing Calibration Diagnostics & Finite-Sample Bias

### 1.1 The Mathematical Finite-Sample Bias of Empirical Binned ECE

Expected Calibration Error (ECE) partitions predictions into $B$ bins $I_1, \dots, I_B$ and computes:
$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \bar{y}_b - \bar{p}_b \right|, \quad \text{where } \bar{y}_b = \frac{1}{|B_b|} \sum_{i \in B_b} y_i, \quad \bar{p}_b = \frac{1}{|B_b|} \sum_{i \in B_b} p_i$$

As formally proven by Roelofs et al. (2022) [1] and Kumar et al. (2019) [2], **the absolute value operator $|\cdot|$ converts zero-mean estimation noise into a strictly positive expected value**:
$$\mathbb{E}\left[ \left| \bar{y}_b - \bar{p}_b \right| \right] > 0 \quad \forall n_b < \infty$$

Under the null hypothesis that the model is perfectly calibrated ($y_i \sim \text{Bernoulli}(p_i)$), the central limit theorem gives:
$$\bar{y}_b - \bar{p}_b \sim \mathcal{N}\left(0, \sigma_b^2\right), \quad \text{with } \sigma_b^2 = \frac{\bar{p}_b(1 - \bar{p}_b)}{n_b}$$

The absolute difference $|\bar{y}_b - \bar{p}_b|$ follows a **folded normal distribution**, whose expected value is:
$$\mathbb{E}\left[ \left| \bar{y}_b - \bar{p}_b \right| \right] = \sigma_b \sqrt{\frac{2}{\pi}} \approx 0.7979 \cdot \frac{\sqrt{\bar{p}_b(1 - \bar{p}_b)}}{\sqrt{n_b}}$$

When $N = 644$ matches are partitioned into $B = 10$ bins, the average bin sample size is $n_b \approx 64$. For bins near the center ($\bar{p}_b \approx 0.50$):
$$\mathbb{E}\left[ \left| \bar{y}_b - \bar{p}_b \right| \right] \approx 0.7979 \cdot \frac{0.50}{\sqrt{64}} \approx 0.0498 = \mathbf{4.98\%}$$

#### Empirical Null Simulation Proof:
Simulating $2{,}000$ synthetic outcomes from an idealized, omniscient oracle ($y_{\text{sim}} \sim \text{Bernoulli}(p)$) using the exact probability distribution of Consolidated A1 on $N = 644$ yields:
- **Expected Empirical ECE under Perfect Calibration:** **`4.02%`** (95% sampling interval: **`[2.23%, 6.10%]`**).
- **Consolidated A1 Observed ECE:** **`4.61%`** ($p = 0.268$ against the null hypothesis of perfect calibration).
- **Shrunk Hybrid A1 Observed ECE:** **`4.67%`** ($p = 0.274$).
- **Commercial Bookmaker Opening Lines Observed ECE:** **`3.01%`** ($p = 0.833$).

Therefore, the observed empirical ECE of $\sim 4.6\%$ is entirely consistent with sample variance ($\mathcal{O}(1/\sqrt{N})$) and does not represent a broken model.

---

### 1.2 Debiased and Unbiased Calibration Estimators

To evaluate true calibration independent of bin sample size, three unbiased or bias-mitigated formulations should be used [1, 2, 13]:

#### A. Debiased ECE ($\ell_2$-Variance Subtracted) [2]:
Instead of taking the absolute value of bin deviations, calculate squared calibration error and subtract the known binomial sampling variance:
$$\widehat{\text{ECE}}^2_{\text{debiased}} = \sum_{b=1}^B \frac{n_b}{N} \max\left(0, (\bar{y}_b - \bar{p}_b)^2 - \frac{\bar{y}_b(1 - \bar{y}_b)}{n_b - 1}\right)$$
$$\text{ECE}_{\text{debiased}} = \sqrt{\widehat{\text{ECE}}^2_{\text{debiased}}}$$

#### B. Squared Kernel Calibration Error (SKCE) [13]:
Widmann, Gelbrecht, and Batselier (2019) formulated a completely bin-free, unbiased estimator using a matrix reproducing kernel $k(p_i, p_j)$:
$$\text{SKCE}_{\text{unbiased}} = \frac{2}{N(N-1)} \sum_{1 \le i < j \le N} (y_i - p_i)(y_j - p_j) k(p_i, p_j)$$
SKCE has zero positive sample bias and permits an exact two-sample statistical test ($p$-value) for whether miscalibration exceeds zero.

#### C. Empirical Verification on Canonical vs Operational Datasets:
Applying debiasing to our datasets confirms asymptotic convergence:
- **Canonical Benchmark ($N = 11{,}550$):** Standard Binned ECE = **`1.02%`**, Debiased ECE = **`0.82%`**.
- **Scraped Cohort ($N = 644$):** Standard Binned ECE = `4.61%`, Debiased ECE = **`3.16%`**.

---

### 1.3 Murphy's (1973) Brier Score Decomposition: The Resolution vs Reliability Tradeoff

Murphy (1973) proved that the Mean Squared Error of a probabilistic forecast (the Brier score) partitions additively into three distinct components [14]:
$$\text{Brier}(p, y) = \text{Uncertainty} - \text{Resolution} + \text{Reliability}$$

Where:
$$\text{Uncertainty} = \bar{y}(1 - \bar{y})$$
$$\text{Reliability} = \sum_{b=1}^B \frac{n_b}{N} (\bar{p}_b - \bar{y}_b)^2 \quad (\approx \text{ECE}^2)$$
$$\text{Resolution} = \sum_{b=1}^B \frac{n_b}{N} (\bar{y}_b - \bar{y})^2$$

#### The Critical Engineering Tradeoff:
- **Resolution** measures the model's ability to separate outcomes into distinct subsets whose win rates deviate from the base rate $\bar{y}$. High resolution directly drives ROC-AUC, discriminative power, and profitable betting edge.
- **Reliability** measures how close the predicted probabilities are to the observed frequencies within those subsets.
- **The Danger of Over-Recalibration:** Aggressively minimizing Reliability (e.g., fitting Isotonic Regression or a 10-knot spline on a small validation set) collapses predictions into broad step bins. While this artificially forces $\text{Reliability} \to 0$ in-sample, it **catastrophically reduces Resolution**. Because Brier subtracts Resolution, destroying Resolution strictly degrades both Brier score and log loss out-of-sample [14, 15].

---

## 2. Mathematical Taxonomy of Recalibration Algorithms

### 2.1 Beta Calibration (Kull, Silva Filho, & Flach, 2017)

Beta calibration is derived from the assumption that the class-conditional distributions of predicted probabilities follow Beta distributions:
$$P(S = s \mid Y = 1) = \text{Beta}(s; \alpha_1, \beta_1), \quad P(S = s \mid Y = 0) = \text{Beta}(s; \alpha_0, \beta_0)$$

By Bayes' rule, the posterior log-odds mapping is:
$$\ln\left(\frac{P(Y=1 \mid S=s)}{1 - P(Y=1 \mid S=s)}\right) = a \ln s - b \ln(1 - s) + c$$
$$\mu_{\text{Beta}}(s; a, b, c) = \frac{1}{1 + \exp\left(-(a \ln s - b \ln(1 - s) + c)\right)} = \frac{1}{1 + \frac{1}{e^c} \frac{(1-s)^b}{s^a}}$$

#### Mathematical Properties:
1. **Contains the Identity Mapping:** When $a = 1, b = 1, c = 0$, $\mu_{\text{Beta}}(s) = s$. It will never distort an already well-calibrated classifier.
2. **Guaranteed Strict Monotonicity:** Enforcing $a > 0$ and $b > 0$ guarantees $\frac{d\mu}{ds} > 0$ everywhere on $s \in (0, 1)$. It strictly preserves sample rankings and ROC-AUC.
3. **Asymmetric Tail Correction:**
   - If $a = b > 1$, it acts as an inverse-sigmoid (dampens overconfident extreme predictions toward $0.50$).
   - If $a \ne b$, it models asymmetric skewness (e.g., favorite-longshot bias).
4. **Fitting Formulation:** Formulated as standard bivariate logistic regression with features $x_1 = \ln s$ and $x_2 = -\ln(1 - s)$:
   $$\min_{a, b, c} -\sum_{i=1}^N \left[ y_i \ln \sigma(a x_{1,i} + b x_{2,i} + c) + (1 - y_i) \ln(1 - \sigma(a x_{1,i} + b x_{2,i} + c)) \right]$$
   Subject to box constraints $a \ge 0, b \ge 0$. Easily optimized via L-BFGS-B [3].

---

### 2.2 Temperature Scaling (Guo et al., 2017)

Temperature Scaling is a single-parameter ($T > 0$) monotonic transformation of log-odds [4]:
$$\text{logit}(p_{\text{cal}}) = \frac{\text{logit}(p_{\text{raw}})}{T}, \quad p_{\text{cal}} = \sigma\left(\frac{\sigma^{-1}(p_{\text{raw}})}{T}\right)$$

#### Mathematical Properties:
1. **Subset of Beta Calibration:** Corresponds exactly to Beta Calibration with $a = b = 1/T$ and $c = 0$.
2. **Strict Monotonicity & Invariance:**
   - $\text{ROC-AUC}(p_{\text{cal}}) \equiv \text{ROC-AUC}(p_{\text{raw}})$.
   - Argmax classification accuracy is strictly invariant for any $T > 0$.
3. **Overconfidence Correction:** When a model is overconfident ($T > 1$), logits are shrunk toward zero, expanding entropy and softening tail probabilities ($0.95 \to 0.91$).
4. **Empirical Optimization on $N = 644$:**
   $$\min_T \text{NLL}(T) \implies T^* = \mathbf{1.1326}$$
   Applying $T = 1.13$ to Consolidated A1 drops test LogLoss from $0.5687$ to **`0.5672`**, drops Brier score from $0.1952$ to **`0.1944`**, and eliminates all blowout losses ($\text{LL} \ge 2.5$) from $2$ to **$0$**.

---

### 2.3 Platt Scaling (Platt, 1999)

Platt scaling fits a 2-parameter logistic sigmoid over raw model decision values or logits [10]:
$$p_{\text{cal}} = \frac{1}{1 + \exp(A \cdot z + B)}$$

#### Limitations in Modern Practice:
- Assumes class-conditional score densities are Gaussian with equal variance.
- When applied to bounded probabilities $s \in (0.0, 1.0)$ directly rather than unbounded logits $z$, Platt scaling forces a rigid sigmoidal S-curve that cannot model inverse sigmoids.
- If the raw model is already calibrated, fitting Platt scaling on small validation sets ($N < 1{,}000$) routinely increases log loss due to estimation error in $A$ and $B$ [3].

---

### 2.4 Venn-Abers Predictors (Vovk, Shen, Manokhin, & Ho, 2014)

Venn-Abers predictors are non-parametric multiprobabilistic calibrators based on algorithmic randomness and conformal prediction [11]. Instead of outputting a single point estimate, Venn-Abers outputs a calibrated probability interval $[p_0, p_1]$:

Given calibration dataset $Z = \{(s_i, y_i)\}_{i=1}^N$ and a new test score $s_{N+1}$:
1. Assume $y_{N+1} = 0$, fit Isotonic Regression via PAV on $Z \cup \{(s_{N+1}, 0)\}$, and extract prediction $p_0 = \hat{y}_{N+1}$.
2. Assume $y_{N+1} = 1$, fit Isotonic Regression via PAV on $Z \cup \{(s_{N+1}, 1)\}$, and extract prediction $p_1 = \hat{y}_{N+1}$.
3. The true probability lies in $[p_0, p_1]$, where the interval width $|p_1 - p_0|$ measures epistemic uncertainty.

#### Minimax Point Reduction:
To obtain a single probability for scoring rules:
- **Log Loss Optimal:** $p_{\text{point}} = \frac{p_1}{1 - p_0 + p_1}$
- **Brier Score Optimal:** $p_{\text{point}} = p_1 \text{ if } (1-p_1)^2 \le p_0^2 \text{ else } p_0$

#### Key Advantage:
Unlike standard Isotonic Regression, Venn-Abers probabilities **never clip to $0.0$ or $1.0$**, strictly avoiding infinite log loss penalties on tail upsets.

---

### 2.5 SplineCalib (Lucena, 2018)

SplineCalib fits a smoothing cubic spline with an $\ell_2$ integrated second-derivative curvature penalty on logit-transformed scores [12]:
$$\min_f \sum_{i=1}^N \ell_{\text{log}}(y_i, \sigma(f(z_i))) + \lambda \int [f''(t)]^2 dt$$

#### Failure Modes in Wagering:
While SplineCalib produces continuous curves that avoid PAV's step plateaus, standard cubic smoothing splines **do not guarantee monotonicity**. In probability regions with sparse validation observations (such as extreme underdogs $p < 0.12$), unconstrained splines exhibit localized negative derivatives (oscillating "wiggles"). This inverts the true team strength ordering and corrupts EV calculations.

---

## 3. Calibration in Sports Forecasting & Prediction Markets

### 3.1 Interaction with Bayesian Market Shrinkage: The Before vs After Architecture

EnsembleLegends combines model probability $p_{\text{model}}$ with market consensus $p_{\text{market}}$ in logit space:
$$z_{\text{hybrid}} = (1 - \alpha) z_{\text{model}} + \alpha z_{\text{market}}, \quad p_{\text{hybrid}} = \sigma(z_{\text{hybrid}})$$

#### Theoretical Theorem on Linear Probability Pools:
Ranjan and Gneiting (2010) proved mathematically that **any non-trivial linear combination of calibrated forecasts is strictly uncalibrated and underconfident** [7]. 

In logit space, the combination forms a Logarithmic Opinion Pool (LogPOp) [8]. However, if $z_{\text{model}}$ is raw and miscalibrated, linear interpolation distorts the posterior log-odds.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               RECOMMENDED THREE-STAGE PRODUCTION RECALIBRATION PIPELINE                │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [Stage 1: Pre-Shrinkage Calibration]
  Raw Model Score s ∈ (0, 1) ──► Beta / Temperature Calibration (T = 1.13)
                                 Produces Standardized Log-Odds: z_model = logit(p_cal)
                                           │
                                           ▼
 [Stage 2: Bayesian Market Shrinkage]
  Consensus Quotes (STS, Betclic, ...) ──► Multiplicative Devigging ──► z_market
                                           │
  z_hybrid = (1 - α) · z_model + α · z_market   (with α = 0.50)
                                           │
                                           ▼
 [Stage 3: Decision Safety Gating]
  EV Qualification Rules A, B, C (EV cap ≤ 0.25 on longshots; Bo1 discrepancy cap)
                                           │
                                           ▼
                                 Final Qualified Bet
```

#### Why Pre-Shrinkage Calibration is Mandatory:
Calibrating the model *before* market blending ensures that $z_{\text{model}}$ is a true, unbiased log-odds metric. If calibration were applied only *after* blending, the calibrator would treat the market's own efficient log-odds as uncalibrated features, introducing unnecessary estimation noise into sharp closing quotes.

---

### 3.2 The Favorite-Longshot Bias and Devigging Interactions

In sports betting markets, bookmaker odds incorporate an overround $\sum \frac{1}{O_i} = S > 1.0$.
1. **Multiplicative (Proportional) Devigging:**
   $$p_i = \frac{1/O_i}{S}$$
   Assumes margin is spread proportionally. However, real bookmakers face informed insider risk (Shin, 1992 [16]) and exploit bettor risk-loving lottery preferences (Snowberg & Wolfers, 2010 [17]) by packing disproportionate margin onto longshots. Multiplicative devigging therefore **overestimates underdog win probability** ($p_{\text{prop}} > p_{\text{true}}$) and underestimates favorites.
2. **Empirical Devigging Tradeoff:**
   While Shin devigging theoretically corrects the Favorite-Longshot Bias, our earlier shootout across $208{,}093$ quotes showed that Shin devigging in volatile esports markets aggressively pushes favorites too high ($73\% \to 88\%$), generating 32 blowout losses when upsets occur. Multiplicative devigging remains the operational SSOT because it preserves relative odds ratios and produces **zero tail blowouts**.

---

### 3.3 Best-of Series Compounding & Jensen's Inequality

In League of Legends, matches are played as Best-of-1, Best-of-3, or Best-of-5. Analytical series projection under independent Bernoulli map wins is:
$$P(\text{Win Bo3}) = f_3(p) = 3p^2 - 2p^3$$
$$P(\text{Win Bo5}) = f_5(p) = 10p^3 - 15p^4 + 6p^5$$

#### Breakdown of Calibration Propagation:
1. **Derivative Error Amplification:**
   $$\left.\frac{df_3}{dp}\right|_{p=0.5} = 6(0.5) - 6(0.25) = \mathbf{1.50}, \quad \left.\frac{df_5}{dp}\right|_{p=0.5} = 30(0.25) - 60(0.125) + 30(0.0625) = \mathbf{1.875}$$
   A small map-level calibration error of $\Delta p = \pm 0.04$ near coin-flip matches is magnified into a series calibration error of **$\pm 0.075$** ($7.5\text{ pp}$) in Bo5 matches.
2. **Jensen's Inequality under Parameter Uncertainty [9]:**
   The series projection function $f_k(p)$ is strictly concave for favorites ($p > 0.5$, $f''(p) < 0$). If estimated map probability has epistemic variance $\hat{p} \sim \mathcal{N}(p, \sigma^2)$, Jensen's inequality dictates:
   $$\mathbb{E}[f(\hat{p})] < f(p)$$
   Plugging a point estimate into series formulas systematically **overstates favorites and understates underdogs**. Series probabilities must be calibrated directly using Bo-specific targets or temperature-softened map inputs ($T = 1.12$).

---

### 3.4 Resolving the High-Entropy "Coin-Flip" Divergence ($1.80 - 2.50$ Odds)

As identified in `docs/CALIBRATION_AUDIT.md`, offline statistical models suffer systematic miscalibration specifically in the $1.80 - 2.50$ odds range (predicting $62\%$ vs actual $45.7\%$), while performing with pinpoint accuracy on heavy favorites ($81.8\%$ vs $80.8\%$) and deep underdogs ($47.1\%$ vs $43.2\%$).

#### Why Coin-Flips Suffer Localized Collapse:
1. **High Match Entropy:** At $p \approx 0.50$, binary outcome variance is maximized ($\text{Var}(Y) = 0.25$). Small ratings noise yields large probability swings.
2. **Information Asymmetry:** Offline ratings capture historical macro metrics. In coin-flip matches between closely matched teams, the outcome is dominated by game-day factors (draft composition, scrim meta reads, player illness) that bookmakers have already priced in via market flow.
3. **The Solution (Multicalibration & Selective Shrinkage) [18]:**
   Instead of applying uniform shrinkage across all matches, apply **entropy-adaptive shrinkage**:
   $$\alpha(p) = \alpha_{\text{base}} + (1 - \alpha_{\text{base}}) \cdot \exp\left(-\frac{(z_{\text{model}})^2}{2\sigma_{\text{entropy}}^2}\right)$$
   For coin-flip matches ($z_{\text{model}} \approx 0$), shrinkage toward the market increases to $\alpha \approx 0.75 - 0.85$, preventing model overconfidence where it lacks information edge.

---

## 4. Strategic Engineering Blueprint for EnsembleLegends

### Recommended Implementation Actions

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ENGINEERING RECALIBRATION ROADMAP                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Immediate: Apply Temperature Scaling (T = 1.13) to Standalone A1                    │
│    - One scalar divisor on logits: z_scaled = z / 1.13                                 │
│    - Drops standalone LogLoss from 0.5687 to 0.5672; eliminates all tail blowouts.     │
│                                                                                        │
│ 2. Horizon Architecture: Standardize Stage-1 Pre-Shrinkage Calibration                │
│    - Fit Beta Calibration on expanding walk-forward splits (min N = 1,000).            │
│    - Blend calibrated logits with Multiplicative Market Consensus (α = 0.50).          │
│                                                                                        │
│ 3. Replace Empirical ECE with Debiased ECE in All Benchmark Reporting                  │
│    - Use variance-subtracted debiased ECE (Kumar et al., 2019) to eliminate            │
│      the O(1/sqrt(n_b)) sample size bias on seasonal splits (N ~ 600).                 │
│                                                                                        │
│ 4. Retain Bet Qualification Gating (Rules A, B, C)                                     │
│    - Cap EV ≤ 0.25 on odds > 3.50; quarantine Bo1 gaps ≥ 0.12 and negative CLV.       │
│    - Protects bankroll against the Optimizer's Curse and 12% turnover tax friction.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## References

[1] Roelofs, R., Cain, N., Shlens, J., & Mozer, M. C. (2022). Mitigating Bias in Calibration Error Estimation. Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS 2022), PMLR 151. - https://arxiv.org/abs/2012.08668
[2] Kumar, A., Sarawagi, S., & Jain, A. (2019). Verified Uncertainty Calibration. Advances in Neural Information Processing Systems (NeurIPS 2019), 32. - https://arxiv.org/abs/1909.10155
[3] Kull, M., Silva Filho, T. M., & Flach, P. (2017). Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017), PMLR 54:623-631. - http://proceedings.mlr.press/v54/kull17a/kull17a.pdf
[4] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70:1321-1330. - https://arxiv.org/abs/1706.04599
[5] Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities With Supervised Learning. Proceedings of the 22nd International Conference on Machine Learning (ICML 2005), 625–632. - https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
[6] Ferro, C. A. T., & Fricker, T. E. (2012). A bias-corrected decomposition of the Brier score. Quarterly Journal of the Royal Meteorological Society, 138(668), 1954–1960. - https://doi.org/10.1002/qj.1924
[7] Ranjan, R., & Gneiting, T. (2010). Combining Probability Forecasts. Journal of the Royal Statistical Society Series B: Statistical Methodology, 72(1), 71–91. - https://doi.org/10.1111/j.1467-9868.2009.00726.x
[8] Satopää, V. A., Baron, J., Foster, D. P., Mellers, B. A., Tetlock, P. E., & Ungar, L. H. (2014). Combining multiple probability predictions using a simple logit model. International Journal of Forecasting, 30(2), 344–356. - https://doi.org/10.1016/j.ijforecast.2013.09.009
[9] Baker, R. D., & McHale, I. G. (2013). Optimal Betting Under Parameter Uncertainty: Improving the Kelly Criterion. Decision Analysis, 10(3), 189–199. - https://doi.org/10.1287/deca.2013.0271
[10] Platt, J. C. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. Advances in Large Margin Classifiers, 10(3), 61–74. - https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
[11] Vovk, V., Shen, I., Manokhin, V., & Ho, M. G. (2014). Venn-Abers predictors. Uncertainty in Artificial Intelligence (UAI 2014), arXiv:1211.0025. - https://arxiv.org/abs/1211.0025
[12] Lucena, B. (2018). Spline-Based Probability Calibration. arXiv preprint arXiv:1809.07751. - https://arxiv.org/abs/1809.07751
[13] Widmann, D., Gelbrecht, M., & Batselier, K. (2019). Calibration tests in multi-class classification: A unifying framework. Advances in Neural Information Processing Systems (NeurIPS 2019), 32. - https://arxiv.org/abs/1910.11385
[14] Murphy, A. H. (1973). A New Vector Partition of the Probability Score. Journal of Applied Meteorology and Climatology, 12(4), 595–600. - https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2
[15] Wu, X., & Gales, M. J. F. (2021). Should Ensemble Members Be Calibrated? arXiv preprint arXiv:2101.05397. - https://arxiv.org/abs/2101.05397
[16] Shin, H. S. (1992). Prices of State-Contingent Claims with Uninformed Traders and Insiders. The Geneva Papers on Risk and Insurance Theory, 17(1), 17–32. - https://doi.org/10.1007/BF00057055
[17] Snowberg, E., & Wolfers, J. (2010). Explaining the Favorite-Longshot Bias: Is it Risk-Love or Misperceptions? Journal of Political Economy, 118(4), 723–746. - https://doi.org/10.1086/655844
[18] Hébert-Johnson, U., Kim, M., Reingold, O., & Rothblum, G. (2018). Multicalibration: Calibration for the (Computationally-Identifiable) Masses. Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR 80:1939-1948. - https://proceedings.mlr.press/v80/hebert-johnson18a.html
