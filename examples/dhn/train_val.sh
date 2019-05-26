#!/bin/bash

# baseline
python -u train_val_script.py

nohup python -u train_val_script.py --gpu 0 --finetune-all False --prefix nofinenute > 1.log 2>&1 &

nohup python -u train_val_script.py --gpu 1 --batch-size 128 --prefix dhn_bt > 2.log 2>&1 &

nohup python -u train_val_script.py --gpu 2 --batch-size 512 --prefix dhn_bt > 3.log 2>&1 &

# nohup python -u train_val_script.py --gpu 3 --alpha 3 > 4.log 2>&1 &
