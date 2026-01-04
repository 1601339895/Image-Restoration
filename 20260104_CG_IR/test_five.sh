#!/bin/bash

set -e  # 遇错退出（可选）


model="model"
de_type="denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie"
benchmarks_list=("derain" "dehaze" "denoise_25" "gopro" "lolv1")
checkpoint_id_base="/data/project/All-in-One_image_restoration/Train/train_log_checkpoints/20251230_experiment/20251225_CG_IR_AGF_useGroupNorm/five_checkpoints/2025_12_30_18_55_13"
data_file_dir="/data/project/All-in-One_image_restoration/img_restoration_data/all_in_one_data"



# =============================
# Test Five: derain / dehaze / denoise25 /gopro /lolv1 
# ==============================

log_dir="/data/project/All-in-One_image_restoration/Train/experiment/20251230_experiment/20251230_CG_IR_AGF_useGroupNorm/test_log"
mkdir -p "$log_dir"

five_log="${log_dir}/epoch_120_20251225_CG_IR_AGF_useGroupNorm_five_test_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Test Five..." | tee "$five_log"

for benchmarks in "${benchmarks_list[@]}"; do
    {
        echo "========================"
        echo "Running: --benchmarks $benchmarks"
        echo "Start time: $(date)"
        echo "------------------------"
    } | tee -a "$five_log"

    python src/test.py \
        --model "$model" \
        --benchmarks $benchmarks \
        --checkpoint_id "${checkpoint_id_base}" \
        --data_file_dir "${data_file_dir}" \
        --de_type $de_type 2>&1 | tee -a "$five_log" \


    {
        echo "End time: $(date)"
        echo ""
    } | tee -a "$five_log"
done

echo "Test Five completed. Log saved to $five_log"
