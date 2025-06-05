#!/bin/bash
#
#SBATCH --array=1-500%40
#SBATCH --cpus-per-task=21
#SBATCH -t 4:00:00
#SBATCH -J corr_perc_moments_%A
#SBATCH -p secondary
#SBATCH -o corr_perc_moments_out_%A_%a.o%j
#SBATCH -e corr_perc_moments_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10

linear_size=8192
gamma=2
threads=21

export PYTHONPATH='/projects/illinois/eng/physics/dahmen/mullen/helper_modules':$PYTHONPATH
python corr_perc_moments.py "$SLURM_ARRAY_TASK_ID" "$linear_size" "$gamma" "$threads"