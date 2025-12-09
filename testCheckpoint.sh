#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --account=COMS033444

module load languages/python/tensorflow-2.16.1
echo "Start"
# Edit to whatever hyperparameters/checkpoint you use
python test_checkpoint.py --checkpoint-name MrCNN_bs=256_lr=0.01_momentum=0.9_dropout=0.2_weightdecay=0.0002_checkpoints/MrCNN_bs=256_lr=0.01_momentum=0.9_dropout=0.2_weightdecay=0.0002_checkpoint_16
echo "Done"
