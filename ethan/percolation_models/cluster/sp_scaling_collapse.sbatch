#!/bin/bash
#
#SBATCH --mem=120000
#SBATCH -t 04:00:00
#SBATCH -J scaling_collapse_siteperc_%A
#SBATCH -p secondary
#SBATCH -o scaling_collapse_siteperc_out_%A_%a.o%j
#SBATCH -e scaling_collapse_siteperc_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: p_c is 0.405
module load anaconda3/2024.02-Feb
python3 scaling_collapse_siteperc.py