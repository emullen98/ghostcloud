#!/bin/bash
#
#SBATCH --mem=96000
#SBATCH -t 04:00:00
#SBATCH -J make_plots_integrated_%A
#SBATCH -p secondary
#SBATCH -o make_plots_integrated_out_%A_%a.o%j
#SBATCH -e make_plots_integrated_err_%A_%a.e%j
#SBATCH --mail-user=emullen2@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# NOTE: %A = $SLURM_ARRAY_JOB_ID
# NOTE: %a = $SLURM_ARRAY_TASK_ID
# NOTE: These two symbols only seem to work when I call them in the # SBATCH section above
# NOTE: critical bond probability is 0.381
python3 make_plots_integrated.py