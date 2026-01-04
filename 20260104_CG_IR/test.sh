#!/bin/bash

trainsets=(
    CDD11_low
    CDD11_haze
    CDD11_rain
    CDD11_snow
    CDD11_low_haze
    CDD11_low_rain
    CDD11_low_snow
    CDD11_haze_rain
    CDD11_haze_snow
    CDD11_low_haze_snow
    CDD11_low_haze_rain
)

model="model"
checkpoint_id="/data/project/All-in-One_image_restoration/experiment/20251208_cpMoCEIR_train_Restoration_Lorain/checkpoints/2025_12_07_23_56_58"
benchmarks="cdd11"
de_type="denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie"
data_file_dir="/data/project/img_restoration_data/CDD11"

log_file="test_results_$(date +%Y%m%d_%H%M%S).log"

for trainset in "${trainsets[@]}"; do
    {
        echo "========================"
        echo "Testing with trainset: $trainset"
        echo "Start time: $(date)"
        echo "------------------------"
    } | tee -a "$log_file"

    python src/test.py \
        --model "$model" \
        --checkpoint_id "$checkpoint_id" \
        --trainset "$trainset" \
        --benchmarks "$benchmarks" \
        --de_type $de_type \
        --data_file_dir "$data_file_dir" 2>&1 | tee -a "$log_file"

    {
        echo "End time: $(date)"
        echo ""
    } | tee -a "$log_file"
done

echo "All tests completed. Results saved to $log_file"