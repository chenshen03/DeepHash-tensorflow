#!/bin/bash

# baseline
python -u train_val_script.py

# nohup python -u train_val_script.py --gpu 0 --cq-lambda 0.01 > 1.log 2>&1 &

# nohup python -u train_val_script.py --gpu 1 --cq-lambda 0.5 > 2.log 2>&1 &

# nohup python -u train_val_script.py --gpu 2 --cq-lambda 0.1 --alpha 5 > 3.log 2>&1 &

# nohup python -u train_val_script.py --gpu 3 --alpha 3 > 4.log 2>&1 &
