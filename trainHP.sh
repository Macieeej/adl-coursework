#!/bin/bash

#SBATCH --job-name=hp_search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o ./output%A_%a/log_%A_%a.out
#SBATCH -e ./output%A_%a/log_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --account=COMS033444
#SBATCH --array=0-624

module load languages/python/tensorflow-2.16.1

# 625 permutations of hyperparameters
WEIGHTDECAY=(0.0001 0.0002 0.0003 0.0004 0.0005)
LEARNINGRATE=(0.002 0.004 0.006 0.008 0.01)
DROPOUT=(0.1 0.2 0.3 0.4 0.5)
#MOMENTUM=(0.9 0.8 0.7 0.6 0.5)

IDX=$SLURM_ARRAY_TASK_ID

# this blew my mind ngl
WD=${WEIGHTDECAY[$((IDX % 5))]}
LR=${LEARNINGRATE[$(((IDX / 5) % 5))]}
DO=${DROPOUT[$(((IDX / 25) % 5))]}
#MOM=${MOMENTUM[$(((IDX / 125) % 5))]}

# change this to the directory where you want to save the checkpoints
CHECKPOINTLOGDIR=MrCNN_bs_256_lr_${LR}_momentum_09_dropout_${DO}_weight_decay_${WD}_checkpoints

echo "Starting run with weight decay: $WD, learning rate: $LR, dropout: $DO"
python mr_cnn.py --weight-decay $WD --learning-rate $LR --dropout $DO

BESTCHECKPOINT=$?
echo "Best checkpoint found was $BESTCHECKPOINT"

python test_checkpoint.py --checkpoint-name $CHECKPOINTLOGDIR/MrCNN_bs_256_lr_${LR}_momentum_0.9_dropout_${DO}_weight_decay_${WD}_checkpoint_${BESTCHECKPOINT}
