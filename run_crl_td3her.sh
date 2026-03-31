#!/bin/bash
#SBATCH --account=mwmeng99
#SBATCH --partition=gpu_mig40
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=jaxgcrl_baselines
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Running on node: $SLURMD_NODENAME"
echo "Job started at: $(date)"

source ~/.bashrc
conda activate jaxgcrl
cd /JaxGCRL-master || exit 1
mkdir -p logs
mkdir -p checkpoints

############################
# CRL experiments
############################

for seed in {0..9}; do
    echo "Starting CRL experiments for seed ${seed} at $(date)"

    # reacher
    jaxgcrl crl \
        --env reacher \
        --num-envs 1024 \
        --energy-fn l2 \
        --contrastive-loss-fn sym_infonce \
        --policy-lr 0.0006 \
        --checkpoint-logdir checkpoints/crl_reacher_seed${seed} \
        --exp-name crl_reacher_seed${seed} \
        --wandb-group crl_reacher \
        --seed ${seed}

    # humanoid
    jaxgcrl crl \
        --env humanoid \
        --num-envs 512 \
        --energy-fn l2 \
        --contrastive-loss-fn sym_infonce \
        --policy-lr 0.0006 \
        --checkpoint-logdir checkpoints/crl_humanoid_seed${seed} \
        --exp-name crl_humanoid_seed${seed} \
        --wandb-group crl_humanoid \
        --seed ${seed}

    # ant_u_maze
    jaxgcrl crl \
        --env ant_u_maze \
        --num-envs 1024 \
        --energy-fn l2 \
        --contrastive-loss-fn sym_infonce \
        --episode-length 1000 \
        --checkpoint-logdir checkpoints/crl_ant_u_maze_seed${seed} \
        --exp-name crl_ant_u_maze_seed${seed} \
        --wandb-group crl_ant_u_maze \
        --seed ${seed}

    # ant_ball
    jaxgcrl crl \
        --env ant_ball \
        --num-envs 1024 \
        --energy-fn l2 \
        --contrastive-loss-fn sym_infonce \
        --policy-lr 0.0006 \
        --checkpoint-logdir checkpoints/crl_ant_ball_seed${seed} \
        --exp-name crl_ant_ball_seed${seed} \
        --wandb-group crl_ant_ball \
        --seed ${seed}
done

############################
# TD3 + HER experiments
############################

for seed in {0..9}; do
    echo "Starting TD3+HER experiments for seed ${seed} at $(date)"

    # reacher
    jaxgcrl td3 \
        --env reacher \
        --min-replay-size 1000 \
        --discounting 0.99 \
        --episode-length 1000 \
        --unroll-length 62 \
        --num-envs 1024 \
        --use-her \
        --checkpoint-logdir checkpoints/td3_her_reacher_seed${seed} \
        --exp-name td3_reacher_seed${seed} \
        --wandb-group td3_reacher \
        --seed ${seed}

    # humanoid
    jaxgcrl td3 \
        --env humanoid \
        --min-replay-size 1000 \
        --discounting 0.99 \
        --episode-length 1000 \
        --unroll-length 62 \
        --num-envs 512 \
        --use-her \
        --checkpoint-logdir checkpoints/td3_her_humanoid_seed${seed} \
        --exp-name td3_humanoid_seed${seed} \
        --wandb-group td3_humanoid \
        --seed ${seed}

    # ant_u_maze
    jaxgcrl td3 \
        --env ant_u_maze \
        --min-replay-size 1000 \
        --discounting 0.99 \
        --episode-length 1000 \
        --unroll-length 62 \
        --num-envs 1024 \
        --use-her \
        --checkpoint-logdir checkpoints/td3_her_ant_u_maze_seed${seed} \
        --exp-name td3_ant_u_maze_seed${seed} \
        --wandb-group td3_ant_u_maze \
        --seed ${seed}

    # ant_ball
    jaxgcrl td3 \
        --env ant_ball \
        --min-replay-size 1000 \
        --discounting 0.99 \
        --episode-length 1000 \
        --unroll-length 62 \
        --num-envs 1024 \
        --use-her \
        --checkpoint-logdir checkpoints/td3_her_ant_ball_seed${seed} \
        --exp-name td3_ant_ball_seed${seed} \
        --wandb-group td3_ant_ball \
        --seed ${seed}
done

echo "All experiments finished at: $(date)"