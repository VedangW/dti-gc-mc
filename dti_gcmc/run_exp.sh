#!/bin/bash

# Command to run dataset_2
python train_mini_batch.py -st stratified_with_weights -e 25 -ucw -cw 1. 25. -act relu6 -ds 1234 --testing