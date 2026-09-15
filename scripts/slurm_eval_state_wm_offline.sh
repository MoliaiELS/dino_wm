#!/bin/bash
# Re-run offline prediction/ranking after evaluation code changes.

#SBATCH --job-name=phase2-offline-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

variant="${1:?dataset variant is required}"
seed="${2:?training seed is required}"
run_group="${3:?run group is required}"
output_name="${4:-evaluation_stratified.json}"
dataset_name="${PHASE2_DATASET_NAME:-pusht_recovery_phase1_pilot_v1}"

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh
export SDL_VIDEODRIVER=dummy

run_dir="$DATASET_DIR/phase2_runs/$run_group/$variant/seed_$seed"
python -m phase2.evaluate_checkpoint \
  --dataset-dir "$DATASET_DIR/$dataset_name" \
  --checkpoint "$run_dir/checkpoint_best.pt" \
  --output "$run_dir/$output_name"
