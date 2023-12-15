#!/bin/bash
echo $(pwd)
export PYTHONPATH=${PYTHONPATH}:$(pwd)

#Load Anaconda Modules
module load anaconda/2023a

python -c "import torch; print(torch.cuda.device_count())"
# python scripts/train.py
# --gpu-device -1
python scripts/train.py
#python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_simclr.json --gpu-device -1