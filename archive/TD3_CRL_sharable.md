# TD3 + CRL: Implementation Overview

> **Subtask 3 — Chunyu & Jeongbin**
> File: `jaxgcrl/agents/td3_crl/td3_crl.py`
> Based on JaxGCRL ([github.com/MichalBortkiewicz/JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL))

---

## 1. How the TD3 Baseline Works in JaxGCRL

### Architecture (`jaxgcrl/agents/td3/`)

TD3 maintains four networks (two live, two target):

| Network | Input | Output | Role |
|---|---|---|---|
| Policy `π` | obs = `[state ‖ goal]` | action ∈ `[−1,1]^A` | **Deterministic** actor |
| Target policy `π_target` | obs | action | Polyak copy of π |
| Q-network (twin) | obs + action | scalar × 2 | Value estimation |
| Target Q (twin) | obs + action | scalar × 2 | Stable Bellman target |

The Q-network is Brax's standard `make_q_network` (twin MLP, `relu`):
```
Q(s, a, g) = MLP([s ‖ g ‖ a]) → scalar     # heads 1 & 2
```

The policy is a deterministic MLP with a `tanh` output layer (bounded `[−1,1]`).  
At inference, clipped Gaussian noise is added for exploration (`exploration_noise=0.4`).

### Training Loop

```
For each gradient step:
  1. Update twin Q-networks (Bellman backup with target smoothing):
       ã = π_target(s') + clip(𝒩(0, σ), −c, c)     ← target policy smoothing
       y = r + γ * min(Q1_target, Q2_target)(s', ã)
       L_critic = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

  2. [Every policy_delay=2 steps] Update actor:
       L_actor = −E[ Q1(s, π(s)) ]                   ← deterministic policy gradient

  3. [Every policy_delay=2 steps] Soft-update targets:
       π_target ← τ*π + (1-τ)*π_target
       Q_target ← τ*Q + (1-τ)*Q_target
```

### Three Stabilisation Tricks in TD3 (vs. DDPG)

| Trick | Description |
|---|---|
| **Twin critics** | Takes the min of two Q-values to reduce overestimation bias |
| **Delayed policy update** | Actor updated every `policy_delay` critic steps to prevent divergence |
| **Target policy smoothing** | Adds noise to target actions before Bellman backup to smooth the Q-surface |

### Key Hyperparameters (from project reproduction)
```
discounting      = 0.99
num_envs         = 1024
unroll_length    = 62     (UTD ratio 1:16)
min_replay_size  = 1000
use_her          = True
policy_delay     = 2
smoothing_noise  = 0.2
noise_clip       = 0.5
exploration_noise = 0.4
```

### Why TD3 Struggles on Goal-Conditioned Tasks

Same root problem as SAC:
- **Sparse reward** → long Q-value bootstrapping chain with almost no signal.
- HER partially compensates but Q-values are still learned from sparse `r ∈ {0,1}`.
- The deterministic policy has no exploration mechanism beyond additive noise.

---

## 2. How CRL Works (recap)

See SAC_CRL sharable for full detail. Key points:

- Replaces Q-network with dual encoder: `φ(s,a) = sa_encoder(s‖a)`, `ψ(g) = g_encoder(g)`.
- Q-value = `energy(φ, ψ)` (e.g. `−‖φ−ψ‖²`).
- Critic trained via symmetric InfoNCE — **no Bellman backup, no reward signal**.
- Goals sampled as future states from the same trajectory (`γ^(t'−t)` weighting).
- Actor: SAC-style entropy-regularized stochastic Gaussian policy.

---

## 3. Our Modification: TD3_CRL

### Key Idea

**TD3_CRL = CRL's contrastive critic + TD3's deterministic actor + delayed updates + target smoothing**

We keep CRL's contrastive dual encoder and InfoNCE critic loss completely unchanged. We replace CRL's stochastic Gaussian actor with TD3's deterministic actor and add TD3's three stabilisation mechanisms — adapted to work with a representation-based (not Bellman-based) critic.

### The Three TD3 Adaptations for CRL

#### 1. Deterministic Actor (no entropy)

CRL's actor loss: `L = E[ α*log π(a|s,g) − energy(φ(s,a_sample), ψ(g)) ]` (stochastic, entropy term)

TD3_CRL's actor loss:
```
action = tanh(MLP(s ‖ g))          ← deterministic, no log_std head
L_actor = −E[ energy(φ(s, π(s)), ψ(g)) ]     ← no entropy term
```

The actor directly maximises the energy between its action's representation and the goal representation.

Exploration during *data collection* uses separate additive Gaussian noise:
```python
action_collect = clip(π(s) + 𝒩(0, exploration_noise), −1, 1)
```

#### 2. Delayed Policy Updates

The actor and its target are updated every `policy_delay=2` critic steps:

```python
# In JAX — branch on gradient step parity (JIT-compatible)
training_state = jax.lax.cond(
    gradient_steps % policy_delay == 0,
    do_actor_and_target_update,   # actor grad step + Polyak target update
    skip_actor_update,            # no-op
    training_state
)
```

This gives the critic more training signal before the actor commits to an update, preventing early over-fitting to a noisy contrastive landscape.

#### 3. Target Policy Smoothing (adapted for CRL)

In vanilla TD3, target smoothing regularises the Bellman target by evaluating the target Q at a noisy version of the target policy's action.

In TD3_CRL, there is no Bellman backup. We adapt the idea differently: the critic is asked to match **not only** the original `(state, action)` pairs, but also a **secondary InfoNCE loss** where the "action" comes from the Polyak-averaged target actor with added noise:

```python
# Primary InfoNCE loss (same as CRL)
sa_repr   = sa_encoder([state ‖ action])
g_repr    = g_encoder(goal)
L_primary = sym_infonce(energy(sa_repr, g_repr))

# Target-smoothing auxiliary InfoNCE loss
ã = clip(π_target(future_state) + clip(𝒩(0,σ), −c, c), −1, 1)
sa_repr_t = sa_encoder([future_state ‖ ã])
L_smooth  = 0.5 * sym_infonce(energy(sa_repr_t, g_repr))

L_critic  = L_primary + L_smooth
```

This forces the encoder to be smooth around the target actor's action manifold — it cannot concentrate energy on a single isolated action.

### Architecture Diagram

```
Observation: [state (state_size) | goal (goal_size)]
                      │
          ┌───────────┴────────────────┐
          │                            │
    ┌─────▼──────────┐         ┌───────▼──────┐
    │ DeterministicActor│       │   Encoders    │
    │ MLP[256×4]+tanh  │       │               │
    │ → action ∈[-1,1] │       │ sa_encoder    │ Input: [state ‖ action]
    └─────┬────────────┘        │ MLP[256×4]    │ → φ ∈ ℝ^64
          │                     │               │
    Polyak copy                 │ g_encoder     │ Input: goal
          │                     │ MLP[256×4]    │ → ψ ∈ ℝ^64
    ┌─────▼────────────┐        └───────┬───────┘
    │ Target actor π̄   │                │
    │ (for smoothing)  │                │
    └──────────────────┘                │
                                        │
                          energy(φ, ψ) = −‖φ−ψ‖²
                          (Q-value, no Bellman backup)
```

### What Changed vs. TD3 Baseline

| Component | TD3 | TD3_CRL |
|---|---|---|
| Critic | Twin Q-network MLP → scalar (Bellman) | Dual encoder → representation (InfoNCE) |
| Critic loss | TD error with target smoothing | Sym-InfoNCE + target-smoothing aux loss |
| Target Q network | Yes (Bellman target) | No (not needed) |
| Target actor | Yes (policy smoothing in Bellman) | Yes (smoothing aux loss in InfoNCE) |
| Reward signal | Sparse `r ∈ {0,1}` | Not used in critic |
| Goal for training | Original goal (or HER) | CRL future-state sampling |
| Policy type | Deterministic | Deterministic (same) |
| Delayed update | Yes (`policy_delay=2`) | Yes (`policy_delay=2`, same) |
| Exploration | Additive noise on action | Additive noise on action (same) |

### What Changed vs. CRL Baseline

| Component | CRL | TD3_CRL |
|---|---|---|
| Policy type | Stochastic Gaussian | Deterministic (`DeterministicActor`) |
| Temperature α | Learned (SAC-style entropy) | **Removed** |
| Actor loss | `α*log π − energy` | `−energy` only |
| Policy update frequency | Every step | Every `policy_delay=2` steps |
| Target actor | No | Yes (Polyak copy, `τ=0.005`) |
| Critic loss | Primary InfoNCE only | Primary + 0.5 × target-smoothing InfoNCE |
| Exploration | Stochastic sampling | Explicit Gaussian noise (`exploration_noise=0.1`) |

### New `DeterministicActor` Network

```python
class DeterministicActor(nn.Module):
    action_size: int
    network_width: int = 256
    network_depth: int = 4        # 4 hidden layers (matches CRL encoder depth)
    ...

    def __call__(self, x):
        for i in range(self.network_depth):
            x = Dense(network_width)(x)
            x = normalize(x)       # optional LayerNorm
            x = activation(x)      # swish (default) or relu
            ...                    # skip connections
        return tanh(Dense(action_size)(x))  # bounded [-1, 1]
```

Compare to CRL's `Actor`, which has two output heads `(mean, log_std)`. `DeterministicActor` has only one head and applies `tanh` directly.

### Recommended Hyperparameters

```bash
jaxgcrl td3_crl --env reacher \
  --num-envs 1024 \
  --energy-fn l2 \
  --contrastive-loss-fn sym_infonce \
  --policy-lr 0.0003 \
  --critic-lr 0.0003 \
  --policy-delay 2 \
  --smoothing-noise 0.2 \
  --noise-clip 0.5 \
  --exploration-noise 0.1 \
  --discounting 0.99 \
  --episode-length 1000 \
  --checkpoint-logdir checkpoints/td3_crl_reacher_seed0 \
  --seed 0
```

Ablation values to try:
- `--policy-delay`: `1` (= no delay, like CRL), `2` (TD3 default), `4`
- `--exploration-noise`: `0.05`, `0.1`, `0.2`, `0.4` (original TD3)
- `--smoothing-noise`: `0.0` (disable aux target loss), `0.1`, `0.2`

---

## 4. File Structure

```
jaxgcrl/agents/td3_crl/
├── __init__.py              # exports TD3_CRL
└── td3_crl.py               # full implementation (~765 lines)
    ├── DeterministicActor   # tanh-squashed MLP (no log_std)
    ├── flatten_batch()      # CRL future-state goal sampling (unchanged)
    ├── update_critic_td3crl()  # InfoNCE + target-smoothing aux loss
    ├── update_actor_td3crl()   # deterministic policy gradient
    ├── soft_update()        # Polyak averaging for target actor
    ├── TrainingState        # actor_state, target_actor_params, critic_state
    ├── TD3_CRL              # @dataclass with all hyperparams
    └── train_fn()           # training loop with delayed actor updates
```

**Integration** — add to `jaxgcrl/agents/__init__.py`:
```python
from .td3_crl import TD3_CRL
```
Add `TD3_CRL` to the `AgentConfig` Union in `jaxgcrl/utils/config.py`.
