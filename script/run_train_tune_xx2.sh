#!/bin/bash


CUDA_VISIBLE_DEVICES=3 \
python3 rewriting_main.py  \
--do_eval \
--do_train \
--finetune \
--config_train=configs.config_s_fine_cf \
--checkpoint output/0_s_zero/gpt2_model.ckpt \

# --checkpoint output/1_s_fine_xx2_1e-05/model_best.ckpt \
# --do_train \
# --do_test  \
# --roc \
# --do_test  \
# --finetune \
# --checkpoint output/dev/s_x1xx2_train_fine_w0.3_counterfactual/model_best.ckpt \

