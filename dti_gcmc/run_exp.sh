#!/bin/bash

# Command for class weights
python train_mini_batch.py -st stratified -do 0.7 -e 500 -act relu6 -ds 1234 -ucw -acw --write_summary --testing --show_test_results

# python train.py --testing -e 200 -ucw -acw --show_test_results
# Command for oversampling
# python train_mini_batch.py -e 25 -act relu6 -ds 1234 --testing --show_test_results