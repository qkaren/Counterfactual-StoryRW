#!/bin/bash


CUDA_VISIBLE_DEVICES=0 \
python3 rewriting_main.py  \
--config_model=configs.config_model_345M \
--finetune \
--do_test  \
--checkpoint output/remove_space/train_m_fine_cf/model_best.ckpt \
--config_train=configs.config_m_fine_cf


# --do_eval \
# --do_train \
#--do_test  \
# --do_eval \
# --do_train \
# --checkpoint output/1_m_zero/gpt2_model.ckpt \
# --checkpoint output/1_m_zero/gpt2_model.ckpt \
# --finetune \
# --checkpoint output/3_m_fine_xx2_1e-05/model_best.ckpt \

# --checkpoint output/dev/m_x1xx2_train_fine_w0.3_counterfactual/model_best.ckpt \