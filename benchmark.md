Our experiments use JaxGCRL suite of simulated environments described in Section 4.2. We evaluate
algorithms in an online setting, with a UTD ratio 1:16 for CRL, TD3, TD3+HER, SAC, SAC+HER,
and 1 : 5 for PPO. We use a batch size of 256 and a discount factor of 0.99 for all methods except
PPO, for which we use a discount factor of 0.97. For every environment, we sample evaluation
goals from the same distribution as training ones and use a replay buffer of size 10M for CRL, TD3,
TD3+HER, SAC, and SAC+HER. We use 1024 parallel environments for all methods except for
PPO, where we use 4096 parallel environments to collect data. All experiments are conducted for 50
million environment steps.

### Table 1: Environments details
| Parameter | Value |
|-----------|-------|
| num_timesteps | 50,000,000 |
| max_replay_size | 10,000 |
| min_replay_size | 1,000 |
| episode_length | 1,000 |
| discounting | 0.99 |
| num_envs | 1024 (512 for humanoid) |
| batch_size | 256 |
| multiplier_num_sgd_steps | 1 |
| action_repeat | 1 |
| unroll_length | 62 |
| policy_lr | 6e-4 |
| critic_lr | 3e-4 |
| contrastive_loss_function | symmetric_infonce |
| energy_function | L2 |
| logsumexp_penalty | 0.1 |
| hidden layers (for both encoders and actor) | [256,256] |
| representation dimension | 64 |

### Table 2: Hyperparameters
| Environment | Goal distance | Termination | Brax pipeline | Goal sampling |
|-----------|-------|-------|-------|-------|
| Reacher | 0.05 | No | Spring | Disc with radius sampled from [0.0, 0.2] |
| Half-Cheetah | 0.5 | No | MJX | Fixed goal location |
| Pusher Easy | 0.1 | No | Generalized | x coordinate sampled from [−0.55, −0.25], y coordinate sampled from [−0.2, 0.2] |
| Pusher Hard | 0.1 | No | Generalized | x coordinate sampled from [−0.65, 0.35], y coordinate sampled from [−0.55, 0.45] |
| Humanoid | 0.51 | Yes | Spring | Disc with radius sampled from [1.0, 5.0] |
| Ant | 0.5 | Yes | Spring | Circle with radius 10.0 |
| Ant Maze | 0.5 | Yes | Spring | Maze specific |
| Ant Soccer | 0.5 | Yes | Spring | Circle with radius 5.0 |
| Ant Push | 0.5 | Yes | MJX | Two possible locations with uniform noise added |

### BENCHMARK PARAMETERS
The parameters used for benchmarking experiments can be found in Table 2.
min_replay_size is a parameter that controls how many transitions per environment should be
gathered to prefill the replay buffer.
max_replay_size is a parameter that controls how many transitions are maximally stored in replay
buffer per environment.