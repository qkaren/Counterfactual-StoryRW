#!/bin/bash

CUDA_VISIBLE_DEVICES=0  \
python3 rewriting_main_fast.py  \
--do_test \
--finetune
#--do_train \
#--do_eval \
#  --checkpoint output/0_recon_pretrain_w1_1_lr_0.0001_tau_0.1/model.ckpt-2800


  #--do_train \
  #--do_eval \
