#!/bin/bash
# Usage: sbatch ... scripts/slurm_eval_visual_probe_suite.sh <seed> <run_group>

#SBATCH --job-name=p3-probe-suite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

seed="${1:?seed is required}"
run_group="${2:?run group is required}"

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh

if [[ "$seed" == "0" ]]; then
  bash scripts/slurm_eval_visual_probes.sh raw_dino 0 "$run_group"
  bash scripts/slurm_eval_visual_probes.sh oracle_state 0 "$run_group"
fi
bash scripts/slurm_eval_visual_probes.sh random_adapter "$seed" "$run_group" D_SFN_balanced
bash scripts/slurm_eval_visual_probes.sh trained_adapter "$seed" "$run_group" D_SFN_balanced
bash scripts/slurm_eval_visual_probes.sh trained_adapter "$seed" "$run_group" D_SFR_balanced
