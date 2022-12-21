#!/bin/bash
#SBATCH --job-name=Animals_svm
#SBATCH --nodes=1 -p classc -A sxg125_csds438
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --time=20:00:00
#SBATCH --mail-user=ech89@case.edu
#SBATCH --mail-type=end
#SBATCH -o /mnt/pan/courses/sxg125_csds438/group7/svmModel/results_svmModelSuperSmall_88.txt

module load gcc/6.3.0
module load python/3.6.6
module load cuda/11.2

#pip3.6 install scikit-image --user
#pip3.6 install opencv-python==4.3.0.36 --user
#pip3.6 install scikit-learn --user
#pip3.6 install pillow==8.3.2 --user
#pip3.6 install scipy --user
#pip3.6 install pandas --user
#pip3.6 install matplotlib --user

echo 0: running the python file 120

python3.6 ./svmModelSuperSmall_120.py
