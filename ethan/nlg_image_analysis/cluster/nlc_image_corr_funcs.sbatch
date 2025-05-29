#!/bin/bash
#
#SBATCH --array=1-404%40
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -t 24:00:00
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
thresh_min=1
thresh_max=100
thread_count=32

python nlc_image_corr_funcs.py "$SLURM_ARRAY_TASK_ID" "$fill_holes" "$frac" "$thresh_min" "$thresh_max" "$thread_count"