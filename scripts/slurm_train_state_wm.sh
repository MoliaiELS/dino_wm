#!/bin/bash
# Submit with, for example:
#   sbatch --output="$DATASET_DIR/logs/phase2-%x-%j.out" \
#     scripts/slurm_train_state_wm.sh D_SFR_balanced 0 phase2_pilot_v1

#SBATCH --job-name=phase2-state-wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

variant="${1:?dataset variant is required}"
seed="${2:?training seed is required}"
run_group="${3:-phase2_state_wm_v1}"
normalization_variant="${4:-$variant}"
dataset_name="${PHASE2_DATASET_NAME:-pusht_recovery_phase1_pilot_v1}"

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh
export SDL_VIDEODRIVER=dummy

output_dir="$DATASET_DIR/phase2_runs/$run_group/$variant/seed_$seed"
python train_state_wm.py \
  --dataset-dir "$DATASET_DIR/$dataset_name" \
  --variant "$variant" \
  --normalization-variant "$normalization_variant" \
  --seed "$seed" \
  --output-dir "$output_dir"

python evaluate_state_wm.py \
  --dataset-dir "$DATASET_DIR/$dataset_name" \
  --checkpoint "$output_dir/checkpoint_best.pt" \
  --output "$output_dir/evaluation.json"
