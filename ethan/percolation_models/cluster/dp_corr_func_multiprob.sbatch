#!/bin/bash
#
#SBATCH --array=1-15%15
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH -t 04:00:00
#SBATCH -J dp_corr_func_%A
#SBATCH -p secondary
#SBATCH -o dp_corr_func_out_%A_%a.o%j
#SBATCH -e dp_corr_func_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
# NOTE: critical bond probability is 0.381
module load anaconda3/2024.06-Jun
size=5000
endTime=7
fillHoles=1
frac=0.01
python dp_corr_func_multiprob.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$size" "$endTime" "$fillHoles" "$frac"