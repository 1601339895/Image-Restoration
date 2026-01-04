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
cdd11_checkpoint_id="/data/project/All-in-One_image_restoration/Train/train_log_checkpoints/20251230_experiment/20251230_CG_IR_AGF_useGroupNorm/CDD11_checkpoints/2026_01_02_20_59_35"
cdd11_benchmarks="cdd11"
cdd11_de_type="denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie"
data_file_dir="/data/project/All-in-One_image_restoration/img_restoration_data/CDD11_data/CDD11"

log_dir="/data/project/All-in-One_image_restoration/Train/experiment/20251230_experiment/20251230_CG_IR_AGF_useGroupNorm/test_log"
mkdir -p "$log_dir"


cdd11_log="${log_dir}/CDD11_epoch_200_test_$(date +%Y%m%d_%H%M%S).log"
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
        --dim 32 \
        --context_dim 64 \
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
