#!/bin/bash
#SBATCH -J GF_RGCC
#SBATCH -p wzhcnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --array=1-25
#SBATCH -o slurm_logs/GF_RGCC_%A_%a.out
#SBATCH -e slurm_logs/GF_RGCC_%A_%a.err

module load apps/python/3.8.10

cd "/work/home/acn1eb8nfq/yuezhang46/GFcode/GF实验二：分解模块"
mkdir -p slurm_logs

RUN_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((1000 + RUN_ID))

python3 GF_RGCC.py --run_id ${RUN_ID} --seed ${SEED}