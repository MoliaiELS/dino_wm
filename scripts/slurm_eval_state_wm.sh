#!/bin/bash
# Submit with, for example:
#   sbatch --output="$DATASET_DIR/logs/phase2-eval-%j.out" \
#     scripts/slurm_eval_state_wm.sh D_SFR_balanced 0 pilot_a98096d

#SBATCH --job-name=phase2-state-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

variant="${1:?dataset variant is required}"
seed="${2:?training seed is required}"
run_group="${3:?run group is required}"

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh
export SDL_VIDEODRIVER=dummy

run_dir="$DATASET_DIR/phase2_runs/$run_group/$variant/seed_$seed"
python evaluate_state_wm.py \
  --checkpoint "$run_dir/checkpoint_best.pt" \
  --output "$run_dir/closed_loop.json" \
  --skip-prediction \
  --skip-ranking \
  --closed-loop
