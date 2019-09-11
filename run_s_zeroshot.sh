#!/bin/bash

# middle size
# zero-shot

gpu_id=1

output_dir="output/s_zeroshot"
config_train=config_train_zeroshot

#config_model=configs.config_model_345M
#pretrained_model_dir=gpt2_pretrained_models/model_345M
#pretrain_checkpoint=gpt2_pretrained_models/model_345M/model.ckpt

mkdir -p ${output_dir}
cp $0 ${output_dir}
cp configs/${config_train}.py ${output_dir}

# input: x1xx2
CUDA_VISIBLE_DEVICES=${gpu_id}  \
python3 rewriting_main.py  \
  --config_train=configs.${config_train} \
  --output_dir=${output_dir} \
  --do_test \
  --finetune

