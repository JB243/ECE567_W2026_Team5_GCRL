# Integration Patches for JaxGCRL

After copying the three agent directories into `jaxgcrl/agents/`, apply these two small patches to register the new agents.

---

## 1. `jaxgcrl/agents/__init__.py`

Replace the current file content with:

```python
from .crl import CRL
from .ppo import PPO
from .sac import SAC
from .td3 import TD3
from .sac_crl import SAC_CRL
from .ppo_crl import PPO_CRL
from .td3_crl import TD3_CRL
```

---

## 2. `jaxgcrl/utils/config.py`

Change the import line and `AgentConfig` union:

```python
# Old:
from jaxgcrl.agents import CRL, PPO, SAC, TD3
AgentConfig = Union[CRL, PPO, SAC, TD3]

# New:
from jaxgcrl.agents import CRL, PPO, SAC, TD3, SAC_CRL, PPO_CRL, TD3_CRL
AgentConfig = Union[CRL, PPO, SAC, TD3, SAC_CRL, PPO_CRL, TD3_CRL]
```

---

## 3. CLI usage examples

### SAC + CRL
```bash
jaxgcrl sac_crl --env reacher \
    --num-envs 1024 \
    --energy-fn l2 \
    --contrastive-loss-fn sym_infonce \
    --policy-lr 0.0006 \
    --her-ratio 0.5 \
    --checkpoint-logdir checkpoints/sac_crl_reacher_seed0 \
    --exp-name sac_crl_reacher_seed0 \
    --wandb-group sac_crl_reacher \
    --seed 0
```

### TD3 + CRL
```bash
jaxgcrl td3_crl --env reacher \
    --num-envs 1024 \
    --energy-fn l2 \
    --contrastive-loss-fn sym_infonce \
    --policy-lr 0.0003 \
    --policy-delay 2 \
    --smoothing-noise 0.2 \
    --noise-clip 0.5 \
    --exploration-noise 0.1 \
    --checkpoint-logdir checkpoints/td3_crl_reacher_seed0 \
    --exp-name td3_crl_reacher_seed0 \
    --wandb-group td3_crl_reacher \
    --seed 0
```

### PPO + CRL
```bash
jaxgcrl ppo_crl --env reacher \
    --num-envs 4096 \
    --discounting 0.97 \
    --contrastive-coeff 1.0 \
    --contrastive-lr 0.0003 \
    --repr-dim 64 \
    --energy-fn l2 \
    --contrastive-loss-fn sym_infonce \
    --checkpoint-logdir checkpoints/ppo_crl_reacher_seed0 \
    --exp-name ppo_crl_reacher_seed0 \
    --wandb-group ppo_crl_reacher \
    --seed 0
```
