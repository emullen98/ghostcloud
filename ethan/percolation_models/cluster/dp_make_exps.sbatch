#!/bin/bash
#
#SBATCH --array=1-2%2
#SBATCH --mem=96000
#SBATCH -t 04:00:00
#SBATCH -J make_exps_%A
#SBATCH -p secondary
#SBATCH -o make_exps_out_%A_%a.o%j
#SBATCH -e make_exps_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: critical bond probability for DP with holes filled is 0.381
module load anaconda3/2024.06-Jun
size=50000
python make_exps.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$size"