#!/bin/bash
#
#SBATCH --mem=10000
#SBATCH -t 04:00:00
#SBATCH -J sp_corr_func_combine_%A
#SBATCH -p secondary
#SBATCH -o sp_corr_func_combine_out_%A.o%j
#SBATCH -e sp_corr_func_combine_err_%A.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
module load anaconda3/2024.02-Feb
size=100
jobid=12164093
python sp_corr_func_combine.py "$size" "$jobid"