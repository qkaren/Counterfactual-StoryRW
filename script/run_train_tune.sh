#!/bin/bash


CUDA_VISIBLE_DEVICES=1 \
python3 rewriting_main.py  \
--do_train \
--do_eval \
--config_train=configs.config_s_fine \
--finetune \
--checkpoint output/0_s_zero/gpt2_model.ckpt \

# --finetune \
# --checkpoint output/0_s_zero/gpt2_model.ckpt \
# --checkpoint output/2_s_tune_1e-05/model_best.ckpt \

# --do_train \
# --do_eval \
# --do_test \
# --roc \
# --finetune \


