#!/bin/bash

# small size
# fits to yy | x1x2ysx1xx2

gpu_id=6
is_train=0
test_checkpoint="output/supervised/train_s_supervised/model_best.ckpt"

output_dir="output/s_supervised"
config_train=config_train_supervised

#config_model=configs.config_model_345M
#pretrained_model_dir=gpt2_pretrained_models/model_345M
#pretrain_checkpoint=gpt2_pretrained_models/model_345M/model.ckpt

mkdir -p ${output_dir}
cp $0 ${output_dir}
cp configs/${config_train}.py ${output_dir}

if [ "$is_train" = 1 ]; then ## train

  CUDA_VISIBLE_DEVICES=${gpu_id}  \
  python3 rewriting_main.py  \
    --config_train=configs.${config_train} \
    --output_dir=${output_dir} \
    --do_train \
    --supervised

else ## test

  # input: x1x2ysx1xx2
  CUDA_VISIBLE_DEVICES=${gpu_id}  \
  python3 rewriting_main.py  \
    --config_train=configs.${config_train} \
    --checkpoint=${test_checkpoint} \
    --output_dir=${output_dir} \
    --do_test

  ## input: x1x2
  #CUDA_VISIBLE_DEVICES=${gpu_id}  \
  #python3 rewriting_main.py  \
  #  --config_model=${config_model} \
  #  --pretrained_model_dir=${pretrained_model_dir} \
  #  --config_train=configs.${config_train} \
  #  --test_checkpoint=${test_checkpoint} \
  #  --output_dir=${output_dir} \
  #  --do_test \
  #  --roc

fi
