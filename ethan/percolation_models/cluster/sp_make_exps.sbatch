#!/bin/bash
#
#SBATCH --array=1-500%40
#SBATCH --mem=80000
#SBATCH -t 04:00:00
#SBATCH -J sp_exps_%A
#SBATCH -p secondary
#SBATCH -o sp_exps_out_%A_%a.o%j
#SBATCH -e sp_exps_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: p_c for site percolation using hole filling scheme is ~0.405
module load anaconda3/2024.06-Jun
size=50000
prob=0.405
python make_exps_sp.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$size" "$prob"