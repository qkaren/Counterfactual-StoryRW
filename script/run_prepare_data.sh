#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python3 prepare_data_rewriting.py \
 --data_dir ./data/10w_data_dev_as_test \
 --max_seq_length 128 #150


 #--data_dir ./data/m_data_2 \
