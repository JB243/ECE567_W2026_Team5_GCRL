# PPO + CRL: Implementation Overview

> **Subtask 2 — Jianyu & Henry**
> File: `jaxgcrl/agents/ppo_crl/ppo_crl.py`
> Based on JaxGCRL ([github.com/MichalBortkiewicz/JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL))

---

## 1. How the PPO Baseline Works in JaxGCRL

### Architecture (`jaxgcrl/agents/ppo/`)

PPO uses Brax's built-in `ppo_networks`:

| Network | Input | Output | Role |
|---|---|---|---|
| Policy (`policy_network`) | obs = `[state ‖ goal]` | distribution params `(μ, σ)` | Stochastic Gaussian actor |
| Value (`value_network`) | obs = `[state ‖ goal]` | scalar `V(s,g)` | Baseline for advantage |

Both are plain MLPs with `relu` activations and hidden sizes `(256, 256)`.

### Training Loop (on-policy)

```
For each epoch:
  1. Collect unroll_length steps × num_envs (on-policy rollout)
  2. Compute GAE advantages:
       δt = r_t + γ*V(s_{t+1}) − V(s_t)
       A_t = Σ_{k=0}^{T-t} (γλ)^k * δ_{t+k}
  3. For num_updates_per_batch passes over num_minibatches:
       L_clip = E[ min( ratio*A, clip(ratio, 1±ε)*A ) ]   ← clipped surrogate
       L_value = MSE( V(s), V_target )
       L_entropy = −E[ H(π(·|s)) ]
       L_total = −L_clip + c1*L_value − c2*L_entropy
       Update (policy + value) with Adam
```

where `ratio = π_new(a|s) / π_old(a|s)` and `ε = clipping_epsilon = 0.3`.

### Key Hyperparameters (from project reproduction)
```
discounting      = 0.97
num_envs         = 4096
unroll_length    = 10
batch_size       = 32
num_minibatches  = 16
num_updates_per_batch = 2
entropy_cost     = 1e-4
gae_lambda       = 0.95
clipping_epsilon = 0.3
```

### Why PPO Is the Weakest Baseline

PPO is fundamentally **on-policy**:
1. **No replay buffer** — it cannot revisit past experiences with HER relabeling.
2. **Sparse reward dominates** — even with GAE, the advantage `A_t` is nearly zero everywhere in a sparse-reward environment because `r_t = 0` for most steps.
3. **No goal relabeling** — PPO as implemented uses only the environment's original goal without HER or future-state relabeling.
4. **Sample inefficient** — on-policy methods typically require far more environment interactions than off-policy methods on the same task.

From the JaxGCRL paper: *"PPO consistently fails to learn a useful policy on the more complex environments"* and *"PPO is poorly suited to sparse-reward tasks."*

---

## 2. How CRL Works (recap)

See SAC_CRL sharable for full detail. Key points:

- Replaces the Q-network with a dual encoder: `φ(s,a) = sa_encoder(s‖a)`, `ψ(g) = g_encoder(g)`.
- Critic trained via **symmetric InfoNCE** on positive pairs `(sₜ, aₜ, future_state_{t'})`:
  ```
  L_contrastive = −E[ 2*energy(φᵢ,ψᵢ) − logsumexp_j energy(φᵢ,ψⱼ)
                                         − logsumexp_i energy(φᵢ,ψⱼ) ]
  ```
- **Goals sampled as future states**: for each `(sₜ, aₜ)`, the goal `g` is a future state `s_{t'}` from the same trajectory, sampled with probability `∝ γ^{t'−t}`.
- **No reward signal needed** — the contrastive loss provides self-supervised, dense supervision.
- CRL is **off-policy** (uses a large replay buffer).

---

## 3. Our Modification: PPO_CRL

### Key Idea

**PPO_CRL = PPO (unchanged policy + value training) + CRL InfoNCE auxiliary loss**

PPO_CRL keeps every component of the PPO training loop exactly as-is. It adds a **separate pair of contrastive encoder networks** (`sa_encoder`, `g_encoder`) that are trained in parallel using CRL's InfoNCE loss, applied to the same on-policy rollout batch.

```
total_loss_per_minibatch:
  PPO (policy + value) step  →  unchanged gradient update on PPO params
  CRL auxiliary step         →  separate gradient update on contrastive encoder params
```

The two optimisers are completely independent — the CRL loss cannot destabilise PPO's clipped objective.

### Why This Helps PPO

CRL's contrastive encoders learn a rich **reachability representation** of the observation space:
- `φ(s, a)` encodes "where can I reach from `(s, a)`"
- `ψ(g)` encodes "what does goal `g` look like in representation space"
- `energy(φ, ψ)` measures goal-reachability — a form of dense reward

Even though the PPO policy and value networks do *not* directly use these encoders, two indirect benefits arise:
1. The auxiliary loss provides a **sanity check** that the agent is exploring goal-relevant directions (useful for debugging and later integration).
2. The encoders can be **used to augment the value function** in a follow-up step (e.g. as an additional input or as a shaped reward), which is the natural next experiment to try.

### On-Policy CRL: Within-Rollout Future-State Sampling

The key challenge is that CRL needs `(state, action, future_goal)` triples, which requires trajectory-level data. PPO already collects full rollouts of `unroll_length` steps per environment — we reuse this data.

```python
def sample_contrastive_pairs(rollout_obs, rollout_actions, traj_ids, key):
    # rollout_obs: shape (T, obs_size) — single-env trajectory
    # Same discounted future-state sampling as CRL
    probs[t, t'] = γ^(t'−t) * same_episode(t, t')
    t' = categorical(probs[t, :])

    state[t]  = rollout_obs[t, :state_size]
    action[t] = rollout_actions[t]
    goal[t]   = rollout_obs[t', goal_indices]   # future state as goal
    return state, action, goal
```

For the PPO mini-batch update, we use the simpler form: the goal is already embedded in the observation `[state ‖ goal]`, so we extract `state = obs[:, :state_size]` and `goal = obs[:, state_size:]` directly from the current batch without needing explicit future-state sampling.

### The Modified `sgd_step` Function

```python
def sgd_step(carry, minibatch):
    training_state, key = carry

    # --- Standard PPO update (policy + value networks) ---
    (ppo_loss, ppo_metrics), ppo_params, ppo_opt_state = ppo_update(
        training_state.params,
        training_state.normalizer_params,
        minibatch,
        key_ppo,
        optimizer_state=training_state.optimizer_state,
    )

    # --- CRL auxiliary update (contrastive encoders only) ---
    state  = minibatch.observation[:, :state_size]
    action = minibatch.action
    goal   = minibatch.observation[:, state_size:]

    def crl_loss_fn(c_params):
        sa_repr = sa_encoder.apply(c_params["sa_encoder"], concat(state, action))
        g_repr  = g_encoder.apply(c_params["g_encoder"], goal)
        logits  = energy(sa_repr[:, None, :], g_repr[None, :, :])
        loss    = sym_infonce(logits) + λ * logsumexp_penalty(logits)
        return loss

    crl_loss, crl_grad = value_and_grad(crl_loss_fn)(training_state.contrastive_params)
    c_updates, new_c_opt_state = contrastive_optimizer.update(crl_grad, c_opt_state)
    new_c_params = apply_updates(training_state.contrastive_params, c_updates)

    # Merge state
    new_state = training_state.replace(
        params=ppo_params,
        optimizer_state=ppo_opt_state,
        contrastive_params=new_c_params,
        contrastive_optimizer_state=new_c_opt_state,
    )
    return new_state, {**ppo_metrics, "contrastive_loss": crl_loss}
```

### Architecture Diagram

```
Observation: [state (state_size) | goal (goal_size)]
                      │
         ┌────────────┴────────────┐
         │ PPO networks            │  CRL auxiliary networks
         │                         │
   ┌─────▼──────┐           ┌──────▼────────┐
   │   Policy   │           │  sa_encoder   │  Input: [state ‖ action]
   │  MLP+Gauss │           │  MLP[256×4]   │  → φ ∈ ℝ^64
   │ → (μ,σ,V)  │           │               │
   └─────┬──────┘           │  g_encoder    │  Input: goal
         │                   │  MLP[256×4]   │  → ψ ∈ ℝ^64
         │                   └──────┬────────┘
         │                          │
  PPO loss (clipped                 │
  surrogate + value)         InfoNCE loss on
   ↓ Adam(3e-4)              (state,action,goal)
                              ↓ Adam(3e-4)
                              [separate optimizer]
```

### What Changed vs. PPO Baseline

| Component | PPO | PPO_CRL |
|---|---|---|
| Policy network | Standard PPO MLP | Unchanged |
| Value network | Standard PPO MLP | Unchanged |
| Critic/Q-value | PPO value function V(s) | Unchanged; CRL encoders are *auxiliary* |
| Reward signal | Sparse `r ∈ {0,1}` | Unchanged for policy; CRL uses none |
| Goal for training | Original goal from env | Unchanged for PPO; CRL uses obs-goal extraction |
| Additional networks | — | `sa_encoder` + `g_encoder` (new) |
| Additional loss | — | CRL InfoNCE auxiliary (`contrastive_coeff=1.0`) |
| Additional optimizer | — | Separate Adam for CRL encoders |
| Training state | `(params, normalizer_params, opt_state, env_steps)` | + `(contrastive_params, contrastive_opt_state)` |

### What Changed vs. CRL Baseline

| Component | CRL | PPO_CRL |
|---|---|---|
| Policy type | Stochastic SAC actor (off-policy) | Stochastic PPO actor (on-policy) |
| Advantage estimation | None (uses contrastive Q-value) | GAE (used for PPO objective) |
| Replay buffer | Yes (large off-policy buffer) | **No** (on-policy, uses current rollout) |
| Critic | Primary objective (InfoNCE) | **Auxiliary** objective only |
| Future-state sampling | From replay buffer (large history) | From current rollout (limited horizon) |
| Multi-GPU | No (single device) | Yes (`jax.pmap` over devices) |

### New `PPOCRLTrainingState`

```python
@dataclass
class PPOCRLTrainingState:
    # Standard PPO state
    params: PPONetworkParams            # policy + value params
    normalizer_params: ...              # running obs statistics
    optimizer_state: optax.OptState     # Adam for PPO
    env_steps: jnp.ndarray

    # CRL auxiliary state (new)
    contrastive_params: dict            # {"sa_encoder": ..., "g_encoder": ...}
    contrastive_optimizer_state: ...    # separate Adam for CRL
```

### Recommended Hyperparameters

```bash
jaxgcrl ppo_crl --env reacher \
  --num-envs 4096 \
  --discounting 0.97 \
  --unroll-length 20 \
  --batch-size 256 \
  --num-minibatches 32 \
  --num-updates-per-batch 4 \
  --contrastive-coeff 1.0 \
  --contrastive-lr 0.0003 \
  --repr-dim 64 \
  --energy-fn l2 \
  --contrastive-loss-fn sym_infonce \
  --logsumexp-penalty-coeff 0.1 \
  --checkpoint-logdir checkpoints/ppo_crl_reacher_seed0 \
  --seed 0
```

Ablation values to try:
- `--contrastive-coeff`: `0.0` (= vanilla PPO), `0.5`, `1.0`, `2.0`, `5.0`
- `--contrastive-lr`: `1e-4`, `3e-4`, `1e-3`
- `--repr-dim`: `32`, `64`, `128`

---

## 4. File Structure

```
jaxgcrl/agents/ppo_crl/
├── __init__.py                    # exports PPO_CRL
└── ppo_crl.py                     # full implementation (~635 lines)
    ├── sample_contrastive_pairs() # within-rollout future-state sampling
    ├── contrastive_auxiliary_loss() # InfoNCE loss for aux encoders
    ├── PPOCRLTrainingState        # PPO state + contrastive encoder state
    ├── PPO_CRL                    # @dataclass with all hyperparams
    └── train_fn()                 # pmap-based training loop
        └── sgd_step()             # PPO update + CRL aux update per minibatch
```

**Integration** — add to `jaxgcrl/agents/__init__.py`:
```python
from .ppo_crl import PPO_CRL
```
Add `PPO_CRL` to the `AgentConfig` Union in `jaxgcrl/utils/config.py`.

---

## 5. Potential Next Steps

Since PPO+CRL treats the contrastive encoders as purely auxiliary, the natural follow-up experiments are:

1. **Contrastive-shaped reward**: use `energy(φ(s,a), ψ(g))` as an intrinsic reward added to the sparse reward. This bridges the gap between CRL's dense signal and PPO's policy gradient.
2. **Shared encoder**: share a feature extractor between the PPO policy network and the CRL `sa_encoder`. This lets the contrastive loss directly improve the policy representation.
3. **Contrastive value function**: replace PPO's value `V(s,g)` with `energy(φ(s, mean_π(s)), ψ(g))` to use CRL's reachability estimate as the baseline.
