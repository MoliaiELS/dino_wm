#!/bin/bash
# Usage: sbatch scripts/slurm_cache_pusht_dino.sh <source_dataset> <cache_name> [extra args]

#SBATCH --job-name=p3-dino-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

source_name="${1:?source dataset name is required}"
cache_name="${2:?cache name is required}"
extra_args=("${@:3}")

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh
export SDL_VIDEODRIVER=dummy

python cache_pusht_dino.py \
  --source-dir "$DATASET_DIR/$source_name" \
  --output-dir "$DATASET_DIR/$cache_name" \
  "${extra_args[@]}"
