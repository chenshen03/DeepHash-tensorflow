#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3

# baseline
CUDA_VISIBLE_DEVICES=0 python -u train_val_script.py --gpu 0 > logs/log1 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -u train_val_script.py --learning-rate 0.002 --batch-size 256 --output-dim 64\
#  --cq-lambda 0.0001 --n-subspace 4 --n-subcenter 256 --R 54000 --dataset cifar10 --gpu 0 > logs/log1 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u train_val_script.py --gpu 0 --n-subspace 4 --n-subcenter 256 > logs/log1 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python -u train_val_script.py --gpu 1 --n-subspace 2 --n-subcenter 64 > logs/log2 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python -u train_val_script.py --gpu 2 --n-subspace 3 --n-subcenter 16 > logs/log3 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python -u train_val_script.py --gpu 3 --n-subspace 4 --n-subcenter 8 > logs/log4 2>&1 &
