#!/bin/bash
# Run this script once on the cluster to set up the conda environment.
# Usage: bash scripts/setup_cluster.sh

set -e

echo "=== Setting up JaxGCRL environment ==="

# Load conda (adjust module name if needed on lighthouse)
module load python/3.10 2>/dev/null || true
eval "$(conda shell.bash hook)"

# Create conda env from environment.yml
mkdir -p /scratch/mingyan_root/lalkarmi/envs

conda env create -f environment.yml --prefix /scratch/mingyan_root/lalkarmi/envs/jaxgcrl \
  || conda env update -f environment.yml --prefix /scratch/mingyan_root/lalkarmi/envs/jaxgcrl

conda activate /scratch/mingyan_root/lalkarmi/envs/jaxgcrl

echo "=== Environment setup complete ==="
echo "Test with: conda activate /scratch/mingyan_root/lalkarmi/envs/jaxgcrl && python -c 'import jax; print(jax.devices())'"
