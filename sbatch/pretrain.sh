#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=a100:1
#SBATCH --output /home/ccoulson/ASL-SIMCLR/SIMCLR-ASL/sbatch/logs/pretrain%j.out
#SBATCH --job-name=simclr_pretrain
#SBATCH --qos=dw87
conda activate asl-simclr
nvidia-smi
python train_pretraining.py