#!/bin/bash
#
#SBATCH --array=1-404%40
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -J cloud_perims_areas_%A
#SBATCH -p secondary
#SBATCH -o cloud_perims_areas_out_%A_%a.o%j
#SBATCH -e cloud_perims_areas_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10

fill_holes="True"
thread_count=1

python nlc_image_perim_areas.py "$SLURM_ARRAY_TASK_ID" "$fill_holes" "$thread_count"