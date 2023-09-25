#!/bin/sh
folds=(0 1 2 3 4)

model="rf"

for fold in "${folds[@]}"; do
    python3 train.py --fold "$fold" --model "$model"
done