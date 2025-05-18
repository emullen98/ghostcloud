#!/bin/bash
#
#SBATCH --mem=20000
#SBATCH -t 04:00:00
#SBATCH -J scaling_collapse_fixedtc_%A
#SBATCH -p secondary
#SBATCH -o scaling_collapse_fixedtc_out_%A_%a.o%j
#SBATCH -e scaling_collapse_fixedtc_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
# NOTE: critical bond probability is 0.381
module load anaconda3/2024.02-Feb
python scaling_collapse_fixedtc_moments.py