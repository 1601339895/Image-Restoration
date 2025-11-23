#!/bin/bash

# 设置路径和参数（可根据需要调整）
model="model"
trainset="CDD11_low"
benchmarks="cdd11"
de_types=("denoise_15" "denoise_25" "denoise_50" "dehaze" "derain" "deblur" "synllie")
data_file_dir="/home/aiswjtu/hl/img_data/Data/"
checkpoint_id="/home/aiswjtu/hl/new_image_restoration/baseline_restormer_new_train/20251120_Restormer_FFT_DSConv_Fusion_elementwise_gate_PAFPN_use_frequency_aware_task_aware_fusion/checkpoints/2025_11_20_11_51_08"

# 构建 --de_type 参数字符串
DE_TYPE_ARGS=""
for de_type in "${de_types[@]}"; do
    DE_TYPE_ARGS+="--de_type $de_type "
done

# 执行 Python 脚本
python src/test.py \
    --model "$model" \
    --trainset "$trainset" \
    --benchmarks "$benchmarks" \
    $DE_TYPE_ARGS \
    --data_file_dir "$data_file_dir" \
    --checkpoint_id "$checkpoint_id"