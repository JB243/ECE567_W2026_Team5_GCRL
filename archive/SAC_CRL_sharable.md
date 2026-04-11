# SAC + CRL: Implementation Overview

> **Subtask 1 — Po-yen & Lina**
> File: `jaxgcrl/agents/sac_crl/sac_crl.py`
> Based on JaxGCRL ([github.com/MichalBortkiewicz/JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL))

---

## 1. How the SAC Baseline Works in JaxGCRL

### Architecture (`jaxgcrl/agents/sac/`)

SAC maintains three networks:

| Network | Input | Output | Role |
|---|---|---|---|
| Policy (`policy_network`) | obs = `[state ‖ goal]` | distribution params `(μ, σ)` | Stochastic Gaussian actor |
| Q-network (`q_network`) | obs + action | scalar × 2 critics | Value estimation |
| Temperature `α` | — | scalar | Entropy coefficient |

The Q-network is a **twin MLP** (`n_critics=2`) with `relu` activations and optional LayerNorm:
```
Q(s, a, g) = MLP([s ‖ g ‖ a]) → scalar          # Critic 1
Q(s, a, g) = MLP([s ‖ g ‖ a]) → scalar          # Critic 2 (twin)
```

The policy outputs logits for a `NormalTanhDistribution`, which samples tanh-squashed Gaussian actions.

### Training Loop

```
For each epoch:
  1. Collect unroll_length steps into replay buffer
  2. Sample batch from replay buffer
  3. [Optional] HER relabeling: replace goal with achieved goal at episode end
  4. Update Q-networks (Bellman backup):
       y = r + γ * min(Q1_target, Q2_target)(s', π(s'))
       L_critic = MSE(Q(s,a,g), y)
  5. Update policy:
       L_actor = E[ α*log π(a|s,g) - min(Q1,Q2)(s,a,g) ]
  6. Update temperature α to match target entropy
  7. Soft-update target Q-networks: θ_target ← τ*θ + (1-τ)*θ_target
```

### Key Hyperparameters (from project reproduction)
```
discounting      = 0.99
num_envs         = 1024
UTD ratio        = 1:5
replay_buffer    = 10M steps
learning_rate    = 1e-4
tau              = 0.005
```

### Why SAC Struggles on Goal-Conditioned Tasks

- **Sparse reward** — the agent only gets `r=1` when it reaches the goal within a threshold; otherwise `r=0`. The Q-network receives almost no learning signal early in training.
- **Bootstrapping instability** — the Bellman backup chains sparse rewards back through many steps, making the Q-function hard to learn with sparse signal.
- **HER partially helps** but only with final-step goal relabeling, leaving most transitions with zero reward.

---

## 2. How CRL Works in JaxGCRL

### Core Idea

CRL replaces the scalar Q-network entirely. Instead of `Q(s,a,g) → scalar`, it learns:

```
φ(s, a) = sa_encoder(s ‖ a)   ∈ ℝ^d      # state-action representation
ψ(g)    = g_encoder(g)         ∈ ℝ^d      # goal representation

Q(s, a, g) ≈ energy(φ(s,a), ψ(g))        # similarity as Q-value
```

where `energy` is one of: `−‖x−y‖₂` (L2), `−‖x−y‖` (norm), `x·y` (dot), or cosine.

### Contrastive (InfoNCE) Critic Loss

For a batch of transitions `{(sᵢ, aᵢ, gᵢ)}`, form the logit matrix:
```
Lᵢⱼ = energy(φ(sᵢ, aᵢ), ψ(gⱼ))          # (B×B) matrix

Symmetric InfoNCE:
L_critic = −E[ 2*Lᵢᵢ − logsumexp_j(Lᵢⱼ) − logsumexp_i(Lᵢⱼ) ]
         + λ * E[ logsumexp_j(Lᵢⱼ)² ]    # logsumexp penalty
```

Diagonal entries are the positive pairs (matching state-action with its own goal); off-diagonal are negatives.

### Goal Sampling: Discounted Future-State

CRL does **not** use the environment's sparse reward. Instead, the goal for each transition `(sₜ, aₜ)` is sampled as a **future state** `s_{t'}` from the same trajectory, with probability proportional to `γ^{t'−t}`:

```python
probs[t, t'] = γ^(t'−t)  if t' > t and same_episode(t, t')
             = 0           otherwise

t' ~ Categorical(probs[t, :])
goal = obs[t'][goal_indices]
```

This gives dense training signal: nearly every transition has a valid contrastive goal.

### Actor Loss (SAC-style)

```
L_actor = E[ α * log π(a|s,g) − energy(φ(s, a_sample), ψ(g)) ]
```

Same entropy regularization as SAC, but the "Q-value" is now the energy function.

---

## 3. Our Modification: SAC_CRL

### Key Idea

**SAC_CRL = CRL's contrastive critic + CRL's future-state goal sampling + HER goal mixing**

We keep CRL's entire network architecture and loss functions unchanged. The single addition is a `her_ratio` parameter that mixes two goal-relabeling strategies:

| Strategy | Source | Goal for transition `t` |
|---|---|---|
| **CRL** (`1 − her_ratio`) | Discounted future state | `obs[t'][goal_indices]` where `t' ~ γ^(t'−t)` |
| **HER** (`her_ratio`) | Achieved goal at episode end | `obs[truncation_step][goal_indices]` |

### Why This Helps

CRL's future-state sampling gives **dense intermediate supervision** — every step has a goal it could reach if it just kept going. HER's achieved-goal relabeling **anchors learning to actually-reached states** — it guarantees some fraction of goals are reachable (since the agent literally reached them). Mixing both widens goal coverage without sacrificing CRL's dense signal.

### Modified `flatten_batch`

```python
# CRL branch: sample future state proportional to γ^(t'−t)
probs = is_future_mask * γ^(t'−t) * same_episode_mask + ε*I
goal_crl = obs[sample_future_index(probs)][:, goal_indices]

# HER branch: use goal achieved at episode truncation
goal_her = obs[truncation_step][:, goal_indices]

# Mix per-sample (independently for each transition)
use_her   = Uniform(0,1) < her_ratio
goal      = where(use_her, goal_her, goal_crl)
```

### Architecture Diagram

```
Observation: [state (state_size) | goal (goal_size)]
                      │
          ┌───────────┴───────────┐
          │                       │
    ┌─────▼──────┐         ┌──────▼──────┐
    │   Actor    │         │  Encoders   │
    │ (Gaussian) │         │             │
    │ MLP[256×2] │         │ sa_encoder  │  Input: [state ‖ action]
    │ → (μ, σ)  │         │ MLP[256×2]  │  → φ ∈ ℝ^64
    └─────┬──────┘         │             │
          │                │ g_encoder   │  Input: goal
          │ sample action  │ MLP[256×2]  │  → ψ ∈ ℝ^64
          │                └──────┬──────┘
          │                       │
          └───────────┬───────────┘
                      │
              energy(φ, ψ) = −‖φ−ψ‖²    ← Q-value (no Bellman)
```

### What Changed vs. SAC Baseline

| Component | SAC | SAC_CRL |
|---|---|---|
| Critic | Twin MLP → scalar (Bellman backup) | Dual encoder → representation (InfoNCE) |
| Critic loss | TD error (MSE) | Symmetric InfoNCE |
| Goal for training | From replay buffer (original or HER) | CRL future-state + HER mix |
| Reward signal | Sparse `r ∈ {0,1}` | Not used in critic |
| Target networks | Yes (for Q) | No (not needed without Bellman) |
| Entropy reg. | Yes (α) | Yes (α, identical) |

### What Changed vs. CRL Baseline

| Component | CRL | SAC_CRL |
|---|---|---|
| Goal sampling | Pure CRL (future-state discounted) | Mixed: CRL + HER (`her_ratio=0.5`) |
| HER support | No | Yes (`her_ratio` ∈ [0,1]) |
| Everything else | — | Identical |

### Recommended Hyperparameters

```bash
jaxgcrl sac_crl --env reacher \
  --num-envs 1024 \
  --energy-fn l2 \
  --contrastive-loss-fn sym_infonce \
  --policy-lr 0.0006 \
  --critic-lr 0.0003 \
  --her-ratio 0.5 \
  --discounting 0.99 \
  --episode-length 1000 \
  --checkpoint-logdir checkpoints/sac_crl_reacher_seed0 \
  --seed 0
```

Ablation values to try for `--her-ratio`: `0.0` (= CRL), `0.3`, `0.5`, `0.7`, `1.0` (= HER only).

---

## 4. File Structure

```
jaxgcrl/agents/sac_crl/
├── __init__.py          # exports SAC_CRL
└── sac_crl.py           # full implementation (~671 lines)
    ├── flatten_batch()  # HER + CRL mixed goal relabeling
    ├── TrainingState    # actor_state, critic_state, alpha_state
    ├── SAC_CRL          # @dataclass with all hyperparams
    └── train_fn()       # training loop (mirrors CRL's loop)
```

**Integration** — add to `jaxgcrl/agents/__init__.py`:
```python
from .sac_crl import SAC_CRL
```
Add `SAC_CRL` to the `AgentConfig` Union in `jaxgcrl/utils/config.py`.
