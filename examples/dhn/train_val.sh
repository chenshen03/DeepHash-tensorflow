#!/bin/bash

export PYTHONPATH=~/Projects/Hash/DeepHash/CY-DeepHash/DeepHash:$PYTHONPATH

lr=0.00005 
q_lambda=0.0
alpha=10.0 # 0.2
dataset=cifar10 # imagenet # cifar10,  nuswide_81, coco
log_dir=tflog
data_root=/home/chenshen/Projects/Hash/DeepHash/CY-DeepHash/data/cifar10

if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

export TF_CPP_MIN_LOG_LEVEL=3
#                                                         lr  output  iter    q_lamb    alpha     dataset     gpu    log_dir
CUDA_VISIBLE_DEVICES=$gpu python -u train_val_script.py      $lr 32     2000    $q_lambda  $alpha   $dataset    0   $log_dir  $data_root
