#!/bin/bash


CUDA_VISIBLE_DEVICES=2 \
python3 rewriting_main.py  \
--finetune \
--do_eval \
--do_train \
--config_train=configs.config_m_fine \
--config_model=configs.config_model_345M \
--checkpoint output/1_m_zero/gpt2_model.ckpt \

# --roc \

# --do_test  \

# --checkpoint output/1_m_zero/gpt2_model.ckpt \
# --checkpoint output/4_m_tune_1e-05/model_best.ckpt \

#--do_test  \