#!/bin/bash


CUDA_VISIBLE_DEVICES=3 \
python3 rewriting_main.py  \
--do_test \
--config_train=configs.config_dev_gen \
--checkpoint output/supervised/train_m_supervised/model_best.ckpt \
--config_model=configs.config_model_345M \
--supervised
# --checkpoint output/remove_space/train_s_fine/model_best.ckpt \
# --finetune \
# --checkpoint output/1_m_zero/gpt2_model.ckpt \
# --finetune \
# --finetune \
# --roc \
# --checkpoint output/0_s_zero/gpt2_model.ckpt \
# --checkpoint output/2_s_tune_1e-05/model_best.ckpt \
# --finetune \
# --do_train \
# --do_eval \

# --checkpoint output/1_m_zero/gpt2_model.ckpt \
# --finetune \
# --do_train \
# --do_eval \
# --checkpoint output/remove_space/train_m_recon_cf/model_best.ckpt \


