#!/bin/bash

# Submit with an explicit output path, for example:
# sbatch --partition=GPU --cpus-per-task=4 --mem=8G --time=02:00:00 \
#   --output=/mnt/slurmfs-4090node3/user_data/yguo704/dino_wm_dataset/logs/phase1-%j.out \
#   scripts/slurm_generate_pusht_phase1.sh

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dino_wm
source bash.sh
export SDL_VIDEODRIVER=dummy

python generate_pusht_phase1.py "$@"
