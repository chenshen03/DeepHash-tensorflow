#!/bin/bash

# export TF_CPP_MIN_LOG_LEVEL=3

# baseline
# CUDA_VISIBLE_DEVICES=0 python -u train_val_script.py --gpu 0

CUDA_VISIBLE_DEVICES=0 python -u train_val_script.py --gpu 0 --alpha 0.2
