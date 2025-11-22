#!/bin/bash
#
#SBATCH --array=1-400%40
#SBATCH --mem=5000
#SBATCH -t 04:00:00
#SBATCH -J png_pva_boot_fit_%A
#SBATCH -p secondary
#SBATCH -o /dev/null
#SBATCH -e png_dist_boot_fit_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10
source ~/.bashrc
conda activate ~/myenv
export PYTHONPATH='/projects/illinois/eng/physics/dahmen/mullen/':$PYTHONPATH  # for accessing helper_scripts

# Check for help option or missing argument
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ -z "$1" ]; then
    echo "Usage: sbatch nlc_png_distributions_bootstrap.sbatch <bootstrap_runs> <thresh_type> <hours>"
    echo ""
    echo "bootstrap_runs formatted as int."
    echo ""
    echo "thresh_type must be either 'argmax' or 'all'."
    echo ""
    echo "hours must be one of 'multi', '0300-0400', '0400-0500', '0500-0600', '0700-0800', '0800-0900', '0900-1000', or '1100-1200'."
    echo ""
    echo "Example: sbatch nlc_png_distributions_bootstrap.sbatch 25 argmax multi"
    exit 0
fi
# Check if bootstrap_runs is a valid integer
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: bootstrap_runs must be a positive integer (got: '$1')"
    echo "Example: 10, 25, 50"
    exit 1
fi  
# Check if thresh_type is valid
if [ "$2" != "argmax" ] && [ "$2" != "all" ]; then
    echo "Error: thresh_type must be either 'argmax' or 'all' (got: '$2')"
    exit 1
fi  
# Check if hours is valid
valid_hours=("multi" "0300-0400" "0400-0500" "0500-0600" "0700-0800" "0800-0900" "0900-1000" "1100-1200")
if [[ ! " ${valid_hours[@]} " =~ " $3 " ]]; then
    echo "Error: hours must be one of 'multi', '0300-0400', '0400-0500', '0500-0600', '0700-0800', '0800-0900', '0900-1000', or '1100-1200' (got: '$3')"
    exit 1
fi

# Update job name with parameters
scontrol update jobid=$SLURM_ARRAY_JOB_ID jobname="nlc_png_bootstrap_bootruns=${1}_threshtype=${2}_hours=${3}_%A"

python /projects/illinois/eng/physics/dahmen/mullen/ghostcloud/ethan/nlc_image_analysis/cluster/nlc_png_distributions_bootstrap.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$1" "$2" "$3"