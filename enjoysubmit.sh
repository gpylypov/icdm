#!/bin/bash

#Slurm Options
#SBATCH -o current_output.out-%j
#SBATCH -c 32
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --mem=70000
#SBATCH --gres=gpu:volta:1

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Initialize Modules
# source ../etc/profile
export PYTHONPATH=${PYTHONPATH}:$(pwd)

#Load Anaconda Modules
module load anaconda/2023a

cd $HOME/icdm

python -c "import torch; print(torch.cuda.device_count())"
python enjoy.py --model=./models/RNNMountaingrid-run.nn 
# --gpu-device -1
#python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_simclr.json --gpu-device -1