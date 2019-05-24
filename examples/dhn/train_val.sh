#!/bin/bash

# baseline
python -u train_val_script.py

nohup python -u train_val_script.py --gpu 0 --output-dim 32 > 1.log 2>&1 &

nohup python -u train_val_script.py --gpu 1 --output-dim 48 --prefix dhn_nobalanced > 2.log 2>&1 &

nohup python -u train_val_script.py --gpu 2 --dataset nuswide_21 > 3.log 2>&1 &

# nohup python -u train_val_script.py --gpu 3 --alpha 3 > 4.log 2>&1 &
