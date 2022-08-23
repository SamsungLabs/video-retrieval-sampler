#!/usr/bin/env bash

GPUS=$1

# Main Results
# MSRVTT
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --no_policy
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.03
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.03 --diff
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/msrvtt-jsfusion.json --freeze_cnn --uniform_weight 0.03 --concat

# DiDeMo
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/didemo.json --no_policy --num_epochs 5
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/didemo.json --freeze_cnn --uniform_weight 0.03 --num_epochs 5
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/didemo.json --freeze_cnn --uniform_weight 0.03 --diff --num_epochs 5
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/didemo.json --freeze_cnn --uniform_weight 0.03 --concat --num_epochs 5

# ActivityNet
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/activitynet.json --no_policy
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/activitynet.json --freeze_cnn --uniform_weight 0.03 --num_workers 16
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/activitynet.json --freeze_cnn --uniform_weight 0.03 --diff --num_workers 16
python -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config configs/activitynet.json --freeze_cnn --uniform_weight 0.03 --concat --num_workers 16