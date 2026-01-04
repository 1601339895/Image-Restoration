#!/bin/bash

set -e  # 遇错退出（可选）


# ==============================
# Test Three: derain / dehaze / denoise
# ==============================

model="model"
de_type="denoise_15 denoise_25 denoise_50 dehaze derain"
checkpoint_id_base="/data/project/All-in-One_image_restoration/Train/train_log_checkpoints/20251230_experiment/20251230_CG_IR_AGF_useGroupNorm/three_checkpoints/2025_12_31_15_40_36"
data_file_dir="/data/project/All-in-One_image_restoration/img_restoration_data/all_in_one_data"


log_dir="/data/project/All-in-One_image_restoration/Train/experiment/20251230_experiment/20251230_CG_IR_AGF_useGroupNorm/test_log"
mkdir -p "$log_dir"


three_log="${log_dir}/epoch_100_20251230_CG_IR_AGF_useGroupNorm_three_test_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Test Three..." | tee "$three_log"

# 注意：denoise 的三个子任务合并为一个 benchmarks 参数
benchmarks_list=("dehaze" "derain" "denoise_15 denoise_25 denoise_50")

for benchmarks in "${benchmarks_list[@]}"; do
    {
        echo "========================"
        echo "Running: --benchmarks $benchmarks"
        echo "Start time: $(date)"
        echo "------------------------"
    } | tee -a "$three_log"

    python src/test.py \
        --model "$model" \
        --benchmarks $benchmarks \
        --checkpoint_id "${checkpoint_id_base}" \
        --data_file_dir "${data_file_dir}" \
        --de_type $de_type 2>&1 | tee -a "$three_log"

    {
        echo "End time: $(date)"
        echo ""
    } | tee -a "$three_log"
done

echo "Test Three completed. Log saved to $three_log"

