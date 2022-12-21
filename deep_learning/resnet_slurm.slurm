#!/bin/bash
#SBATCH --job-name=CNN_animal10
#SBATCH --mem=8gb
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /mnt/pan/courses/sxg125_csds438/group7/deep_learning/resnet_output.txt

module load gcc/6.3.0
module load python/3.8.6
module load cuda/11.2

#pip3 install -r requirements.txt --user


echo 0: running the python file

python3 ./deep_learning.py -model_name=resnet -epoch=30
