# Deep Research: Inductive & Dynamic Champion Embeddings via Self-Supervised Multi-Task Pretraining

**Date:** 2026-09-18  
**Author:** AI & Esports Modeling Research Group  
**Topic:** Transitioning from Static Transductive ID Embeddings to Patch-Adaptive Inductive Champion Representations in MOBA Match Prediction

---

## 1. The Core Limitation of Static Champion ID Embeddings (`nn.Embedding(256, 32)`)

Traditional competitive League of Legends and MOBA models utilize static ID lookup tables (`nn.Embedding(num_champions, embedding_dim)`). This creates three severe architectural liabilities:
1. **The Patch Drift & Meta Shift Failure:**
   When Riot Games releases a patch (every 14 days) or applies a hotfix rework (e.g. changing an assassin into a tank itemizer), a static ID table retains historical coordinate inertia. The model cannot adapt until thousands of new matches slowly gradient-update the fixed vector.
2. **The Out-of-Vocabulary (OOV) & Rework Cold-Start:**
   When a new champion (e.g. Smolder, Ambessa) or a full VGU rework (e.g. Skarner) is introduced, the model has zero pre-existing vectors. It cannot extrapolate from playstyle and must initialize from an arbitrary random distribution.
3. **Role-Context Blindness:**
   A champion played as a Top Lane weak-side tank (e.g., Gragas Top) is tactically dissimilar to Gragas Jungle (AP burst initiator) or Gragas Support. A single static ID vector conflates these roles into an uninformative average.

---

## 2. The Inductive Paradigm: Champion as a Function of Contextual Map Streams

Instead of treating a champion as an immutable integer ID $i \in \{1 \dots 170\}$, an **Inductive Dynamic Champion Encoder** maps a temporal sequence of the champion's $K$ most recent matches into a latent embedding space:

$$\mathbf{e}_{(\text{champ}, \text{role}, t)} = \mathcal{F}_{\theta}\Big( \{\mathbf{x}_k\}_{k=1}^K, \; \text{Role}, \; \text{Recency}, \; \text{Tier} \Big) \in \mathbb{R}^{d}$$

### Mathematical Properties:
1. **Patch-Invariance:** When a patch drops, $\mathcal{F}_{\theta}$ reads the new match stream from that patch, automatically shifting the champion's embedding to its new functional location (e.g. shifting toward hyper-carry or utility tank).
2. **Inductive Zero-Shot Generalization:** Even for a brand-new or reworked champion, after just 5–10 professional games worldwide, the Transformer encoder attends over those matches and constructs a rich latent vector.

---

## 3. Self-Supervised Pretraining Strategy

To ensure rich, multi-dimensional champion embeddings that do not overfit to match win/loss labels alone, the champion encoder is pretrained independently via **Multi-Task Self-Supervised Pretraining**:

```
[Sequence of Last K Maps for Champion on Role]
                     │
                     ▼
       Transformer / Perceiver Encoder
                     │
                     ▼
           Champion Latent Vector e ∈ ℝ³²
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
Task 1:          Task 2:         Task 3:
Masked Token     Contrastive     Performance
Reconstruction   Patch Pairing   Regression
(Autoencoder)    (SimCLR / CLIP) (WinRate, DPM, Gold)
```

### Pretext Task 1: Masked Feature Autoencoding (Denoising Reconstruction)
- Randomly mask 20% of the boxscore statistics in the historical maps.
- The model reconstructs the missing values conditioned on the latent embedding:
  $$\mathcal{L}_{\text{recon}} = \frac{1}{|\mathcal{M}|} \sum_{j \in \mathcal{M}} \|\hat{\mathbf{x}}_j - \mathbf{x}_j\|^2$$

### Pretext Task 2: Multi-Task Macro & Micro Outcome Head
- Simultaneously predict downstream macro dynamics:
  $$\hat{y}_{\text{win}}, \; \hat{y}_{\text{dpm}}, \; \hat{y}_{\text{gd15}}, \; \hat{y}_{\text{kda}} = \text{Heads}(\mathbf{e})$$
  $$\mathcal{L}_{\text{outcome}} = \text{BCE}(\hat{y}_{\text{win}}, y) + \lambda \text{MSE}(\hat{y}_{\text{stats}}, y_{\text{stats}})$$

### Pretext Task 3: Patch-Neighborhood Contrastive Learning
- Augmented views of the same champion's recent match stream (subsampling subsets of maps, perturbing opponent tiers) form positive pairs $(\mathbf{e}_1, \mathbf{e}_2)$.
- Different champions on the same role form hard negative pairs.
- InfoNCE loss pulls the champion's representations into a cohesive cluster while separating distinct tactical roles:
  $$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{e}_i, \mathbf{e}_i^+) / \tau)}{\sum_j \exp(\text{sim}(\mathbf{e}_i, \mathbf{e}_j) / \tau)}$$

---

## 4. Downstream Integration into A1

In the downstream match prediction model (A1):
- We **completely eliminate** `nn.Embedding(num_champions, 32)`.
- At prediction time, for each player's chosen or primary champion on that role, we feed the champion's rolling match stream through the frozen/fine-tuned `DynamicChampionEncoder`.
- The resulting dynamic vector $\mathbf{e}_{\text{dynamic}} \in \mathbb{R}^{32}$ replaces the categorical ID embedding.
- The Set Transformer then processes these contextualized representations to evaluate team synergy.
