#!/bin/bash

# baseline
python -u train_val_script.py

nohup python -u train_val_script.py --gpu 0 --batch-size 64 --prefix dhn_bt > 1.log 2>&1 &

nohup python -u train_val_script.py --gpu 1 --batch-size 128 --prefix dhn_bt > 2.log 2>&1 &

nohup python -u train_val_script.py --gpu 2 --prefix bias_weights > 3.log 2>&1 &

nohup python -u train_val_script.py --gpu 3 --finetune-all False --prefix nofinetune > 4.log 2>&1 &
