#!/bin/bash
#
#SBATCH -t 04:00:00
#SBATCH -J corr_func_trial_%A
#SBATCH -p secondary
#SBATCH -o corr_func_trial_out_%A_%a.o%j
#SBATCH -e corr_func_trial_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10

python get_corr_funcs_trial.py
