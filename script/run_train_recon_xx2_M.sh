#!/bin/bash


CUDA_VISIBLE_DEVICES=1 \
python3 rewriting_main.py  \
--config_model=configs.config_model_345M \
--checkpoint output/dev/m_x1xx2_train_recon_w0.3_counterfactual/model_best.ckpt \
--do_test  \

# --checkpoint output/5_m_recon_xx2_1e-05/model_best.ckpt \
# --do_eval \
# --do_train \