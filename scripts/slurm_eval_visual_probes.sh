#!/bin/bash
# Usage: sbatch ... <representation> <seed> <run_group> [variant]

#SBATCH --job-name=p3-visual-probes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

representation="${1:?representation is required}"
seed="${2:?seed is required}"
run_group="${3:?run group is required}"
variant="${4:-}"

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh

train_cache="${PHASE3_CACHE_NAME:-pusht_recovery_dino_cache_v2_4x4_b9585d1}"
test_cache="${PHASE3_TEST_CACHE_NAME:-pusht_recovery_dino_cache_confirmatory_seed20260916_60_4x4_b9585d1}"
output_dir="$DATASET_DIR/phase3_runs/$run_group/probes/$representation"
mkdir -p "$output_dir"
args=(
  --train-cache-dir "$DATASET_DIR/$train_cache"
  --test-cache-dir "$DATASET_DIR/$test_cache"
  --representation "$representation"
  --seed "$seed"
  --output "$output_dir/seed_$seed${variant:+_$variant}.json"
)
if [[ "$representation" == "trained_adapter" || "$representation" == "random_adapter" ]]; then
  if [[ -z "$variant" ]]; then
    echo "variant is required for adapter representations" >&2
    exit 2
  fi
  args+=(--checkpoint "$DATASET_DIR/phase3_runs/$run_group/$variant/seed_$seed/checkpoint_best.pt")
fi
python evaluate_visual_probes.py "${args[@]}"
