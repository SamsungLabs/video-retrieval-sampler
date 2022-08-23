#!/usr/bin/env bash

GPUS=$1

# Ablation Studies for MSRVTT
# Changing loss weights
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.3

# Different temporal modelling modules
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.03 --rnn lstm
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.03 --rnn bilstm
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.03 --no_rnn

# Using ViT-B/16
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --no_policy --clip_backbone ViT-B/16 --train_batch_size 16 --val_batch_size 16
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.03 --clip_backbone ViT-B/16 --train_batch_size 16 --val_batch_size 16