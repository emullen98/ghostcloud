#!/bin/bash
#
#SBATCH --mem=10000
#SBATCH -t 04:00:00
#SBATCH -J integrated_dist_%A
#SBATCH -p secondary
#SBATCH -o integrated_dist_out_%A_%a.o%j
#SBATCH -e integrated_dist_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
# NOTE: critical bond probability is 0.381
module load anaconda3/2024.02-Feb
python integrated_dist.py