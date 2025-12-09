#!/bin/bash

#SBATCH --job-name=hp_search_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out
#SBATCH -e ./log_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --account=COMS033444

module load languages/python/tensorflow-2.16.1

python mr_cnn.py

BESTCHECKPOINT=$?
echo "Best checkpoint found was $BESTCHECKPOINT"

python test_checkpoint.py --checkpoint-name MrCNN_bs_256_lr_0.01_momentum_0.9_dropout_0.2_weight_decay_0.0002_checkpoints/MrCNN_bs_256_lr_0.01_momentum_0.9_dropout_0.2_weight_decay_0.0002_checkpoint_${BESTCHECKPOINT}
