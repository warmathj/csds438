#!/bin/bash
#SBATCH --job-name=Animals_KMeans
#SBATCH --nodes=1 -p classc -A sxg125_csds438
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --time=20:00:00
#SBATCH --mail-user=shp68@case.edu
#SBATCH --mail-type=end
#SBATCH -o /mnt/pan/courses/sxg125_csds438/group7/results_kmeansClustering.txt

module load gcc/6.3.0
module load python/3.6.6
module load cuda/11.2

#pip3.6 install scikit-image --user
#pip3.6 install opencv-python==4.3.0.36 --user
#pip3.6 install scikit-learn --user

python3.6 kMeansClustering.py