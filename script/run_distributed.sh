#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 \
mpirun -np 3 \
    -H localhost:3 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl tcp,self \
    -mca btl_tcp_if_include lo \
    python3 rewriting_main.py --do_train \
    --distributed \
    --output_dir=./output


#CUDA_VISIBLE_DEVICES=1 python3 bert_classifier_main.py --do_eval \
#   --config_data=config_data_roc \
#   --output_dir=./output \
#   --checkpoint ./output/model.ckpt
