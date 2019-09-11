#!/bin/bash


CUDA_VISIBLE_DEVICES=3 \
python3 rewriting_main.py  \
--finetune \
--do_test  \
--config_model=configs.config_model_345M \
--checkpoint output/1_m_zero/gpt2_model.ckpt \
# --roc \
# --finetune
# --do_eval \
# --do_test  \
# --do_train \

