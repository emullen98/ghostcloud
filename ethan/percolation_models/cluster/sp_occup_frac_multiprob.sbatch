#!/bin/bash
#
#SBATCH --array=1-100%40
#SBATCH --mem=30G
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -J sp_occup_frac_multiprob_%A
#SBATCH -p secondary
#SBATCH -o sp_occup_frac_multiprob_out_%A_%a.o%j
#SBATCH -e sp_occup_frac_multiprob_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
# NOTE: Site percolation p_c is 0.405 for the hole filling scheme.
module load anaconda3/2024.06-Jun
size=5000
fill_holes=1
python sp_occup_frac_multiprob.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$size" "$fill_holes"
