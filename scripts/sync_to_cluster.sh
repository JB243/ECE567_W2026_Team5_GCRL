#!/bin/bash
# Syncs this repo to the cluster.
# Usage: bash scripts/sync_to_cluster.sh
# Run from the repo root on your local machine.

REMOTE_USER="lalkarmi"
REMOTE_HOST="lighthouse.arc-ts.umich.edu"
REMOTE_DIR="/home/${REMOTE_USER}/JaxGCRL"

echo "=== Syncing to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} ==="

rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='runs/' \
  --exclude='logs/' \
  --exclude='.venv' \
  . "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "=== Sync complete ==="
echo ""
echo "Next steps on the cluster:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  bash scripts/setup_cluster.sh   # first time only"
echo "  mkdir -p logs"
echo "  sbatch scripts/submit_job.sh"
