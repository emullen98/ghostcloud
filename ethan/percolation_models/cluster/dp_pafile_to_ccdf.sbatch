#!/bin/bash
#
#SBATCH --array=1-40%40
#SBATCH --mem=20000
#SBATCH -t 04:00:00
#SBATCH -J dp_ccdfs_over_probs_%A
#SBATCH -p secondary
#SBATCH -o dp_ccdfs_overprobs_out_%A_%a.o%j
#SBATCH -e dp_ccdfs_overprobs_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
module load anaconda3/2024.06-Jun
python dp_pafile_to_ccdf.py "$SLURM_ARRAY_TASK_ID"