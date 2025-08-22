#!/bin/bash
#
#SBATCH --array=1-100%40
#SBATCH -t 4:00:00
#SBATCH -J test_corr_func_%A
#SBATCH -p secondary
#SBATCH -o test_corr_func_out_%A_%a.o%j
#SBATCH -e test_corr_func_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10

python sample.py "$SLURM_ARRAY_TASK_ID"