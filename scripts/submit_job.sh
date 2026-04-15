#!/bin/bash
#SBATCH --job-name=jaxgcrl
#SBATCH --account=mingyan
#SBATCH --partition=mingyan-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   sbatch --job-name=crl_ant_ball scripts/submit_job.sh crl ant_ball
#   sbatch --job-name=sac_her_reacher scripts/submit_job.sh sac_her reacher
#   sbatch --job-name=crl_bigbuf_ant_ball scripts/submit_job.sh crl_bigbuf ant_ball "0 1 2"
#
# METHOD: crl | crl_bigbuf | sac | sac_her | td3 | td3_her | ppo | sac_crl | sac_crl_noher
# ENV:    reacher | humanoid | ant_u_maze | ant_ball
# SEEDS:  (optional) space-separated list, default "0 1 2 3 4 5 6 7 8 9"

set -e

METHOD=${1:-crl}
ENV=${2:-ant_ball}
SEEDS=${3:-"0 1 2 3 4 5 6 7 8 9"}

PYTHON=/scratch/mingyan_root/lalkarmi/envs/jaxgcrl/bin/python

cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints

echo "=== JaxGCRL: method=${METHOD}, env=${ENV} ==="
echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# humanoid needs fewer parallel envs due to memory
if [[ "$ENV" == humanoid* ]]; then
  NUM_ENVS=512
else
  NUM_ENVS=1024
fi

for seed in $SEEDS; do
  CKPT_DIR="checkpoints/${METHOD}_${ENV}_seed${seed}"
  mkdir -p "$CKPT_DIR"
  echo "--- Seed ${seed} ---"

  BASE_ARGS="--env ${ENV} --seed ${seed} --num_envs ${NUM_ENVS}
    --total_env_steps 50000000 --num_evals 500
    --episode_length 1000 --action_repeat 1
    --checkpoint_logdir ${CKPT_DIR}
    --wandb_project_name jaxgcrl
    --wandb_group ${METHOD}_${ENV}
    --exp_name ${METHOD}_${ENV}_seed${seed}
    --log_wandb"

  if [[ "$METHOD" == "crl" ]]; then
    # Use code defaults (norm + fwd_infonce + lr=3e-4), NOT the paper's table values.
    # The paper table values (l2 + sym_infonce + lr=6e-4) cause TD3+HER to outperform CRL.
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py crl $BASE_ARGS \
      --batch_size 256 \
      --unroll_length 62 \
      --min_replay_size 1000 \
      --max_replay_size 10000 \
      --discounting 0.99 \
      --train_step_multiplier 1

  elif [[ "$METHOD" == "crl_bigbuf" ]]; then
    # Same as CRL but with 4x larger replay buffer for more diverse InfoNCE negatives.
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py crl $BASE_ARGS \
      --batch_size 256 \
      --unroll_length 62 \
      --min_replay_size 1000 \
      --max_replay_size 40000 \
      --discounting 0.99 \
      --train_step_multiplier 1

  elif [[ "$METHOD" == "crl_bigbatch" ]]; then
    # Same as CRL but with 4x larger batch size for more InfoNCE negatives per update.
    # More negatives per gradient step = better contrastive representation, no speed penalty.
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py crl $BASE_ARGS \
      --batch_size 1024 \
      --unroll_length 62 \
      --min_replay_size 1000 \
      --max_replay_size 10000 \
      --discounting 0.99 \
      --train_step_multiplier 1

  elif [[ "$METHOD" == "crl_learntemp" ]]; then
    # CRL with learned InfoNCE temperature (CLIP-style).
    # Temperature is jointly optimized with the critic instead of fixed at τ=1.
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py crl $BASE_ARGS \
      --batch_size 256 \
      --unroll_length 62 \
      --min_replay_size 1000 \
      --max_replay_size 10000 \
      --discounting 0.99 \
      --train_step_multiplier 1 \
      --learn_temperature

  elif [[ "$METHOD" == "sac" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py sac $BASE_ARGS \
      --min_replay_size 1000 \
      --discounting 0.99 \
      --unroll_length 62

  elif [[ "$METHOD" == "sac_her" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py sac $BASE_ARGS \
      --min_replay_size 1000 \
      --discounting 0.99 \
      --unroll_length 62 \
      --use_her

  elif [[ "$METHOD" == "td3" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py td3 $BASE_ARGS \
      --min_replay_size 1000 \
      --discounting 0.99 \
      --unroll_length 62

  elif [[ "$METHOD" == "td3_her" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py td3 $BASE_ARGS \
      --min_replay_size 1000 \
      --discounting 0.99 \
      --unroll_length 62 \
      --use_her

  elif [[ "$METHOD" == "sac_crl" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py sac_crl $BASE_ARGS \
      --min_replay_size 1000 \
      --discounting 0.99 \
      --unroll_length 62 \
      --her_ratio 0.5 \
      --energy_fn l2 \
      --contrastive_loss_fn sym_infonce \
      --policy_lr 6e-4 \
      --critic_lr 3e-4

  elif [[ "$METHOD" == "sac_crl_noher" ]]; then
    # her_ratio=0.0: pure CRL goal sampling, no HER mixing.
    # Tests whether SAC entropy regularization alone (vs CRL actor) is the contribution.
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py sac_crl $BASE_ARGS \
      --min_replay_size 1000 \
      --discounting 0.99 \
      --unroll_length 62 \
      --her_ratio 0.0 \
      --energy_fn l2 \
      --contrastive_loss_fn sym_infonce \
      --policy_lr 6e-4 \
      --critic_lr 3e-4

  elif [[ "$METHOD" == "ppo" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl \
    $PYTHON run.py ppo $BASE_ARGS \
      --num_envs 4096 \
      --discounting 0.97
  fi

done

echo "=== All seeds done ==="
