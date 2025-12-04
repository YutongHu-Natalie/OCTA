#!/bin/bash
#SBATCH --job-name=gtssl
#SBATCH --output=../logs/instructerc_%j.out
#SBATCH --error=../logs/instructerc_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mem=60G
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
cd /local/scratch/yhu383/OCTA/Tree_Project
source ../venv/bin/activate
bash run_gtssl_finetune.sh
echo "Job finished at $(date)"
