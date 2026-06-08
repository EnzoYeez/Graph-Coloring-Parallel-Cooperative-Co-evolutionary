#!/bin/bash
#SBATCH -J US_GA
#SBATCH -p wzhcnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=96:00:00
#SBATCH --array=1-25
#SBATCH -o slurm_logs/US_GA_%A_%a.out
#SBATCH -e slurm_logs/US_GA_%A_%a.err

module load apps/python/3.8.10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p slurm_logs

RUN_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((1000 + RUN_ID))

python3 US_GA.py --run_id ${RUN_ID} --seed ${SEED}
