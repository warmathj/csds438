#!/bin/bash

#SBATCH --job-name=Animals_logReg
#SBATCH --nodes=1 -p classc -A sxg125_csds438
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --mail-user=tkv3@case.edu
#SBATCH --mail-type=end
#SBATCH --time=20:00:00
#SBATCH -o /mnt/pan/courses/sxg125_csds438/group7/results_logReg.txt

module load gcc/6.3.0
module load python/3.8.6
module load cuda/11.2

#pip3 install numpy --user
#pip3 install pandas --user
#pip3 install scikit-image --user
#pip3 install scikit-learn --user

python3 logReg.py