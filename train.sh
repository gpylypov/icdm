module load anaconda/2023a
python -c "import torch; print(torch.cuda.device_count())"
python train.py --config=./configs/dev.yaml 