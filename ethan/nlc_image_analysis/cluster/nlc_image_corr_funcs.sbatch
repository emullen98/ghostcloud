#!/bin/bash
#
#SBATCH --array=1-284%40
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH -J cloud_corr_func_%A
#SBATCH --account=dahmen-ic
#SBATCH -p IllinoisComputes
#SBATCH -o cloud_corr_func_out_%A_%a.o%j
#SBATCH -e cloud_corr_func_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10

fill_holes="True"
frac=0.01
thread_count=32

export PYTHONPATH='/projects/illinois/eng/physics/dahmen/mullen/helper_modules':$PYTHONPATH
python nlc_image_corr_funcs.py "$SLURM_ARRAY_TASK_ID" "$fill_holes" "$frac" "$thread_count"