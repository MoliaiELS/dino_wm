#!/bin/bash
# Usage: sbatch scripts/slurm_train_visual_wm.sh <variant> <seed> <run_group> [extra args]

#SBATCH --job-name=p3-visual-wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

variant="${1:?variant is required}"
seed="${2:?seed is required}"
run_group="${3:?run group is required}"
extra_args=("${@:4}")
cache_name="${PHASE3_CACHE_NAME:-pusht_recovery_dino_cache_v2}"

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh

run_dir="$DATASET_DIR/phase3_runs/$run_group/$variant/seed_$seed"
python -m phase3.train \
  --cache-dir "$DATASET_DIR/$cache_name" \
  --variant "$variant" \
  --seed "$seed" \
  --output-dir "$run_dir" \
  "${extra_args[@]}"
