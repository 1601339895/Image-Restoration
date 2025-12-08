#!/bin/bash

# 设置可见的 GPU 设备
export CUDA_VISIBLE_DEVICES=2,3

# 运行训练脚本
#All-in_One Three
# python src/train.py \
#     --model model \
#     --batch_size 8 \
#     --de_type denoise_15 denoise_25 denoise_50 dehaze derain \
#     --trainset standard \
#     --num_gpus 2 \
#     --epochs 200 \
#     --gate_type elementwise \
#     --LoRa_ffn_ratio 0.5 \
#     --LoRa_attn_ratio 0.8 \
#     --data_file_dir /home/aiswjtu/hl/img_data/Data/

#All-in_One Five
python src/train.py \
    --model model \
    --batch_size 8 \
    --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie  \
    --trainset standard \
    --num_gpus 2 \
    --epochs 200 \
    --gate_type elementwise \
    --LoRa_ffn_ratio 0.5 \
    --LoRa_attn_ratio 0.8 \
    --data_file_dir /home/aiswjtu/hl/img_data/Data/


#All-in_One CDD11

# python src/train.py \
#     --model model \
#     --batch_size 8 \
#     --de_type denoise_15 denoise_25 denoise_50 dehaze derain  \
#     --trainset CDD11_all \
#     --num_gpus 2 \
    # --epochs 200 \
    # --gate_type elementwise \
    # --LoRa_ffn_ratio 0.5 \
    # --LoRa_attn_ratio 0.8 \
    # --data_file_dir /home/aiswjtu/hl/img_data/Data/
