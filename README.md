# ECE567_W2026_Team5_GCRL

This repository was created to evaluate the reproducibility of Online Goal-Conditioned Reinforcement Learning.

<p align="center">
  <a href="./ECE567_W2026_Team5_GCRL.pdf">
    <img src="./ECE567_W2026_Team5_GCRL.gif" alt="ECE567 Team 5 CRL Preview" width="750">
  </a>
</p>

---

## Benchmark

* [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) 

## Experimental Setup

Our experiments use JaxGCRL suite of simulated environments. We evaluate algorithms in an online setting, with a UTD ratio 1:16 for CRL, TD3, TD3+HER, SAC, SAC+HER, and 1:5 for PPO. We use a batch size of 256 and a discount factor of 0.99 for all methods except PPO, for which we use a discount factor of 0.97. For every environment, we sample evaluation goals from the same distribution as training ones and use a replay buffer of size 10M for CRL, TD3, TD3+HER, SAC, and SAC+HER. We use 1024 parallel environments for all methods except for PPO, where we use 4096 parallel environments to collect data. All experiments are conducted for 50 million environment steps.

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
| Humanoid | 0.51 | Yes | Spring | Disc with radius sampled from [1.0, 5.0] |
| Ant Maze | 0.5 | Yes | Spring | Maze specific |
| Ant Soccer | 0.5 | Yes | Spring | Circle with radius 5.0 |


## Baselines

* [Contrastive Reinforcement Learning](https://arxiv.org/pdf/2206.07568)

* [SAC](https://arxiv.org/pdf/1801.01290)

* [PPO](https://arxiv.org/pdf/1707.06347)

* [TD3](https://arxiv.org/pdf/1802.09477) 

## Environments

* [Brax](https://github.com/google/brax)

* Locomotion

* Manipulation

## Project Structure

### 📁 **Runs Directory** (Contains all benchmark runs organized by algorithm)
- `runs/` - All benchmark runs organized by algorithm
    - `crl/` - CRL algorithm runs
        - `reacher_runs/` - 10 CRL Reacher benchmark runs
        - `humanoid_runs/` - 10 CRL Humanoid benchmark runs
        - `ant_ball_runs/` - 10 CRL Ant Ball benchmark runs
        - `ant_u_maze_runs/` - 10 CRL Ant U-Maze benchmark runs
    - `ppo/` - PPO algorithm runs
        - `reacher_runs/` - 10 PPO Reacher benchmark runs
        - `humanoid_runs/` - 10 PPO Humanoid benchmark runs
        - `ant_ball_runs/` - 10 PPO Ant Ball benchmark runs
        - `ant_u_maze_runs/` - 10 PPO Ant U-Maze benchmark runs
    - `sac/` - SAC algorithm runs
        - `reacher_runs/` - 10 SAC Reacher benchmark runs
        - `humanoid_runs/` - 10 SAC Humanoid benchmark runs
        - `ant_ball_runs/` - 10 SAC Ant Ball benchmark runs
        - `ant_u_maze_runs/` - 10 SAC Ant U-Maze benchmark runs
    - `sac_her/` - SAC+HER algorithm runs
        - `reacher_runs/` - 10 SAC+HER Reacher benchmark runs
        - `humanoid_runs/` - 10 SAC+HER Humanoid benchmark runs
        - `ant_ball_runs/` - 10 SAC+HER Ant Ball benchmark runs
        - `ant_u_maze_runs/` - 10 SAC+HER Ant U-Maze benchmark runs
    - `td3/` - TD3 algorithm runs
        - `reacher_runs/` - 10 TD3 Reacher benchmark runs
        - `humanoid_runs/` - 10 TD3 Humanoid benchmark runs
        - `ant_ball_runs/` - 10 TD3 Ant Ball benchmark runs
        - `ant_u_maze_runs/` - 10 TD3 Ant U-Maze benchmark runs
    - `td3_her/` - TD3+HER algorithm runs
        - `reacher_runs/` - 10 TD3+HER Reacher benchmark runs
        - `humanoid_runs/` - 10 TD3+HER Humanoid benchmark runs
        - `ant_ball_runs/` - 10 TD3+HER Ant Ball benchmark runs
        - `ant_u_maze_runs/` - 10 TD3+HER Ant U-Maze benchmark runs
  

**Each environment directory contains:**
- CSV files for each seed (s0-s9)
- `metadata.json` in root `runs/` directory contains metadata for all runs


### 📁 **Plot Directory**
- `plots/` - Generated visualization plots
  - IQM with standard error plots (sampled points)
  - Model comparison plots
  - Cross-environment analysis plots

### 📄 **Essential Scripts**
1. `download_all_projects.py` - Downloads all runs from wandb projects
2. `plot_iqm_sampled.py` - Main plotting script for IQM with standard error (samples n points from each run)
3. `plot_all_envs_comparison.py` - Generate plots of different models running in all envs 

## Quick Start

### 1. Download all data:
```bash
python download_all_projects.py
```

**Note:** Modify `download_all_projects.py` with your credentials:
```python
# Specify the projects to download (without username prefix)
PROJECT_NAMES = [
    "jaxgcrl_benchmark_ant",
    "jaxgcrl_benchmark_humanoid", 
    "jaxgcrl_benchmark_ant_ball",
    "jaxgcrl_benchmark_ant_u_maze"
]

USER = "your_wandb_username"  # Replace with your actual wandb username
```

### 2. Generate IQM plots for a single model (50 sampled points):
```bash
python -c "from plot_iqm_sampled import plot_iqm_sampled; import matplotlib.pyplot as plt; fig, ax, data = plot_iqm_sampled('runs/ppo/reacher_runs', 'PPO', 'reacher', n_points=50, success_column='success_rate', save_path='env_plots/ppo_reacher_iqm.png')"
```

### 3. Run complete workflow example:
```bash
python complete_workflow_example.py
```

**Environments for each algorithm:**
- Ant Ball
- Ant U-Maze  
- Humanoid
- Reacher

**Data Structure:**
- 10 seeds per algorithm-environment combination (s0-s9)
- All runs available in `runs/{algorithm}/{environment}/` directories

## License

* MIT License
