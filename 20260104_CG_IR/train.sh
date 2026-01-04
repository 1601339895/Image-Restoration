#!/bin/bash

# 设置可见的 GPU 设备
# export CUDA_VISIBLE_DEVICES=2,3

# 运行训练脚本
#All-in_One Three
# python src/train.py \
#     --model model \
#     --batch_size 8 \
#     --de_type denoise_15 denoise_25 denoise_50 dehaze derain \
#     --trainset standard \
#     --num_gpus 8 \
#     --epochs 120 \
#     --data_file_dir /data/project/All-in-One_image_restoration/img_restoration_data/all_in_one_data/ \
#     --ckpt_dir /data/project/All-in-One_image_restoration/Train/train_log_checkpoints/20251230_experiment/20251225_CG_IR_AGF_useGroupNorm/three_checkpoints 
    # --resume_from /data/project/All-in-One_image_restoration/Train/experiment/20251218_experiment/20251215_cpMoCE_IR_Rectiformer/RectiFormer_three_checkpoints/2025_12_18_20_02_21



#All-in_One Five
python src/train.py \
    --model model \
    --batch_size 16 \
    --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie  \
    --trainset standard \
    --num_gpus 4 \
    --epochs 120 \
    --data_file_dir /data/project/All-in-One_image_restoration/img_restoration_data/all_in_one_data \
    --ckpt_dir /data/project/All-in-One_image_restoration/Train/train_log_checkpoints/20260104_CG_IR/five_checkpoints 





#All-in_One CDD11

# python src/train.py \
#     --model model \
#     --batch_size 16 \
#     --de_type denoise_15 denoise_25 denoise_50 dehaze derain  \
#     --trainset CDD11_all \
#     --num_gpus 4 \
#     --epochs 200 \
#     --dim 32 \
#     --context_dim 64 \
#     --data_file_dir  /data/project/All-in-One_image_restoration/img_restoration_data/CDD11_data/CDD11 \
#     --ckpt_dir /data/project/All-in-One_image_restoration/Train/train_log_checkpoints/20251230_experiment/20251230_CG_IR_AGF_useGroupNorm/CDD11_checkpoints

