#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3

# baseline
CUDA_VISIBLE_DEVICES=1 python -u train_val_script.py > logs/log2 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u train_val_script.py --learning-rate 0.002 --batch-size 256 --output-dim 64\
#  --cq-lambda 0.0001 --n-subspace 4 --n-subcenter 256 --R 54000 --dataset cifar10 --gpu 0 > logs/log1 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u train_val_script.py --gpu 0 --n-subspace 4 --n-subcenter 256 > logs/log1 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python -u train_val_script.py --gpu 1 --n-subspace 2 --n-subcenter 64 > logs/log2 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python -u train_val_script.py --gpu 2 --n-subspace 3 --n-subcenter 16 > logs/log3 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python -u train_val_script.py --gpu 3 --n-subspace 4 --n-subcenter 8 > logs/log4 2>&1 &


# lr=0.002
# q_lambda=0.0001
# subspace_num=4
# dataset=cifar10 # cifar10, nuswide_81
# log_dir=tflog
# data_root=/home/chenshen/Projects/Hash/DeepHash/CY-DeepHash/data/cifar10

# if [ -z "$1" ]; then
#     gpu=0
# else
#     gpu=$1
# fi

# filename="lr_${lr}_cqlambda_${q_lambda}_subspace_num_${subspace_num}_T_${T}_K_${K}_graph_laplacian_lambda_${gl_lambda}_gl_loss_${gl_loss}_dataset_${dataset}"
# model_file="models/${filename}.npy"

# #                               lr    output  iter      q_lamb   n_sub      dataset     gpu      log_dir
# python train_val_script.py      $lr    300     5000    $q_lambda    4       $dataset    $gpu  $log_dir $data_root
