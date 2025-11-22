#!/bin/bash
#
#SBATCH --array=1-400%40
#SBATCH --mem=5000
#SBATCH -t 04:00:00
#SBATCH -J png_pva_boot_fit_%A
#SBATCH -p secondary
#SBATCH -o /dev/null
#SBATCH -e png_pva_boot_fit_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID

module load anaconda3/2024.10

# Check for help option or missing argument
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ -z "$1" ]; then
    echo "Usage: sbatch nlc_png_pva_bootstrap.sbatch <min_area> <logbin_count> <bootstrap_runs> <thresh_type> <hours>"
    echo ""
    echo "min_area, logbin_count, and bootstrap_runs formatted as ints."
    echo ""
    echo "thresh_type must be either 'argmax' or 'all'."
    echo ""
    echo "hours must be one of 'multi', '0300-0400', '0400-0500', '0500-0600', '0700-0800', '0800-0900', '0900-1000', or '1100-1200'."
    echo ""
    echo "Example: sbatch nlc_png_pva_bootstrap.sbatch 2000 10 25 argmax multi"
    exit 0
fi
# Check if min_area is a valid integer
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: min_area must be a positive integer (got: '$1')"
    echo "Example: 1000, 2000, 5000"
    exit 1
fi  
# Check if logbin_count is a valid integer
if ! [[ "$2" =~ ^[0-9]+$ ]]; then
    echo "Error: logbin_count must be a positive integer (got: '$2')"
    echo "Example: 5, 10, 20"
    exit 1
fi  
# Check if bootstrap_runs is a valid integer
if ! [[ "$3" =~ ^[0-9]+$ ]]; then
    echo "Error: bootstrap_runs must be a positive integer (got: '$3')"
    echo "Example: 10, 25, 50"
    exit 1
fi  
# Check if thresh_type is valid
if [ "$4" != "argmax" ] && [ "$4" != "all" ]; then
    echo "Error: thresh_type must be either 'argmax' or 'all' (got: '$4')"
    exit 1
fi  
# Check if hours is valid
valid_hours=("multi" "0300-0400" "0400-0500" "0500-0600" "0700-0800" "0800-0900" "0900-1000" "1100-1200")
if [[ ! " ${valid_hours[@]} " =~ " $5 " ]]; then
    echo "Error: hours must be one of 'multi', '0300-0400', '0400-0500', '0500-0600', '0700-0800', '0800-0900', '0900-1000', or '1100-1200' (got: '$5')"
    exit 1
fi

# Update job name with parameters
scontrol update jobid=$SLURM_ARRAY_JOB_ID jobname="nlc_png_bootstrap_amin=${1}_logbins=${2}_bootruns=${3}_threshtype=${4}_hours=${5}_%A"

python /projects/illinois/eng/physics/dahmen/mullen/ghostcloud/ethan/nlc_image_analysis/cluster/nlc_png_pva_bootstrap.py "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$1" "$2" "$3" "$4" "$5"