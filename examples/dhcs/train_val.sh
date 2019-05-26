#!/bin/bash

# baseline

nohup python -u train_val_script.py --gpu 3 --prefix cosloss_balanced > 4.log 2>&1 &

# nohup python -u train_val_script.py --gpu 1 --cq-lambda 0.5 > 2.log 2>&1 &

# nohup python -u train_val_script.py --gpu 2 --cq-lambda 0.1 --alpha 5 > 3.log 2>&1 &

# nohup python -u train_val_script.py --gpu 3 --prefix dhcs_cos_loss > 4.log 2>&1 &
