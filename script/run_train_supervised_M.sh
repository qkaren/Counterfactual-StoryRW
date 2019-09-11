#!/bin/bash


CUDA_VISIBLE_DEVICES=2 \
python3 rewriting_main.py  \
--do_eval \
--do_train \
--supervised \
--config_model=configs.config_model_345M \
--checkpoint output/1_m_zero/gpt2_model.ckpt \
--config_train=configs.config_m_supervised \
--output_dir=output/supervised
# --checkpoint output/2_s_recon_xx2_1e-05/model_best.ckpt \


# --do_train \
# --do_test  \
# --do_eval \
# --checkpoint 'output/dev/s_train_reconstr_w0.3_counterfactual/model_best.ckpt' \

