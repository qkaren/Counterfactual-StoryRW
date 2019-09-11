#!/bin/bash

# ==============
## 117M model

CUDA_VISIBLE_DEVICES=1  \
python3 rewriting_main.py  \
--do_train \
--do_test \
--finetune

#


# --do_eval \
# --finetune
#--do_train \
#--do_eval \
#  --checkpoint output/0_recon_pretrain_w1_1_lr_0.0001_tau_0.1/model.ckpt-2800

# ===============
## 345M model

# CUDA_VISIBLE_DEVICES=0  \
# python3 rewriting_main.py  \
# --config_model=configs.config_model_345M \
# --pretrained_model_dir=gpt2_pretrained_models/model_345M \
# --pretrain_checkpoint=gpt2_pretrained_models/model_345M/model.ckpt \
# --do_test  #\
