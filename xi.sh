#!/bin/bash
#
#SBATCH --job-name=testname
#SBATCH --partition=q2
#SBATCH --oversubscribe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=v2.o%j
#SBATCH --error=v2.o%j

cd $SLURM_SUBMIT_DIR
echo " "
echo "Job started at `date`"

python3 fhl-home/for_ssh/python_code/plot_xi.py "$@"

echo " "
echo "Job ended at `date`"
echo " "