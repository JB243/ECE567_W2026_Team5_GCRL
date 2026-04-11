#!/bin/bash
# Submit all baseline × environment jobs for the Phase-1 replication.
#
# Baselines: crl, sac, sac_her, td3, td3_her, ppo, sac_crl  (7 methods)
# Environments: reacher, humanoid, ant_u_maze, ant_ball  (4 envs)
# = 28 jobs total, each running 10 seeds on a single A100
#
# Usage:
#   bash scripts/submit_all.sh          # all 28 jobs
#   bash scripts/submit_all.sh sac_crl  # one method across all 4 envs

set -e

METHOD_ARG=${1:-all}

ENVS=(reacher humanoid ant_u_maze ant_ball)

if [[ "$METHOD_ARG" == "all" ]]; then
  METHODS=(crl sac sac_her td3 td3_her ppo sac_crl)
else
  METHODS=("$METHOD_ARG")
fi

mkdir -p logs

for METHOD in "${METHODS[@]}"; do
  for ENV in "${ENVS[@]}"; do
    JOB_NAME="${METHOD}_${ENV}"
    JOB_ID=$(sbatch \
      --job-name=${JOB_NAME} \
      scripts/submit_job.sh ${METHOD} ${ENV} \
      | awk '{print $4}')
    echo "Submitted: ${JOB_NAME} -> job ${JOB_ID}"
  done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u lalkarmi"
