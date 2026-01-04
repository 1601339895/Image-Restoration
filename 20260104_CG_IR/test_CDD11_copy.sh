#!/bin/bash

set -e  # 遇错退出（可选）

# ==============================
# CDD11 Test
# ==============================
trainsets=(
    CDD11_low CDD11_haze CDD11_rain CDD11_snow CDD11_low_haze
    CDD11_low_rain CDD11_low_snow CDD11_haze_rain CDD11_haze_snow
    CDD11_low_haze_snow CDD11_low_haze_rain
)

cdd11_model="model"
cdd11_checkpoint_id="/data/project/All-in-One_image_restoration/Train/train_log_checkpoints/20251225_train_checkpoint/20251225_CG_IR_new_skip/CDD11_checkpoints/2025_12_29_01_44_01"
cdd11_benchmarks="cdd11"
cdd11_de_type="denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie"
data_file_dir="/data/project/All-in-One_image_restoration/img_restoration_data/CDD11"

log_dir="/data/project/All-in-One_image_restoration/Train/experiment/20251225_experiment/20251225_CG_IR_new_skip/test_log"
mkdir -p "$log_dir"


cdd11_log="${log_dir}/epoch_200_CDD11_FFT0.05_test_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Test Three..." | tee "$cdd11_log"


for trainset in "${trainsets[@]}"; do
    {
        echo "========================"
        echo "Testing with trainset: $trainset"
        echo "Start time: $(date)"
        echo "------------------------"
    } | tee -a "$cdd11_log"

    python src/test.py \
        --model "$cdd11_model" \
        --checkpoint_id "$cdd11_checkpoint_id" \
        --trainset "$trainset" \
        --benchmarks "$cdd11_benchmarks" \
        --de_type $cdd11_de_type \
        --data_file_dir "$data_file_dir" 2>&1 | tee -a "$cdd11_log"

    {
        echo "End time: $(date)"
        echo ""
    } | tee -a "$cdd11_log"
done

echo "CDD11 tests completed. Log saved to $cdd11_log"
