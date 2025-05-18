#!/bin/bash
#
#SBATCH --array=1-40%40
#SBATCH --mem=80000
#SBATCH -t 04:00:00
#SBATCH -J sp_pa_over_probs_%A
#SBATCH -p secondary
#SBATCH -o sp_pa_over_probs_out_%A_%a.o%j
#SBATCH -e sp_pa_over_probs_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
# NOTE: critical occupation probability for site percolation with holes filled is 0.405
module load anaconda3/2024.06-Jun
size=50000
python sp_make_pa_over_probs.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$size"