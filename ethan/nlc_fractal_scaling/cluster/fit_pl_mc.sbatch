#!/bin/bash
#
#SBATCH --array=1-500%40
#SBATCH --mem=5000
#SBATCH -t 04:00:00
#SBATCH -J area_perim_pl_fit_%A
#SBATCH -p secondary
#SBATCH -o area_perim_pl_fit_%A_%a.o%j
#SBATCH -e area_perim_pl_fit_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
module load anaconda3/2024.06-Jun
thresh=30G
python fit_pl_mc.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$thresh"