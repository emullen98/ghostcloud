#!/bin/bash
#
#SBATCH --array=1-100%20
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH -t 04:00:00
#SBATCH -J sp_corr_func_%A
#SBATCH -p secondary
#SBATCH -o sp_corr_func_out_%A_%a.o%j
#SBATCH -e sp_corr_func_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
# NOTE: Critical probability for sp with filled holes is 0.405
# NOTE: Critical probability for sp without filled holes is 0.5927
module load anaconda3/2024.02-Feb
size=1000
prob=0.405
fillHoles=1
frac=0.01
python sp_corr_func.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$size" "$prob" "$fillHoles" "$frac"