#!/bin/bash

# ==============================================================================
# Quick Start Training Scripts for Different Datasets
# ==============================================================================

echo "=================================================="
echo "Image Restoration Training - Quick Start Scripts"
echo "=================================================="

# ==============================================================================
# Configuration - MODIFY THESE PATHS
# ==============================================================================
DATA_DIR="/path/to/datasets"  # Change to your dataset directory
CKPT_DIR="checkpoints"
NUM_GPUS=2  # Number of GPUs to use

# ==============================================================================
# Training Script 1: AIO-3 (Three degradations)
# Denoising + Deraining + Dehazing
# ==============================================================================
train_aio3() {
    echo ""
    echo "🚀 Training AIO-3 Model (Denoise + Derain + Dehaze)"
    echo "=================================================="

    python train_improved.py \
        --model model_improved \
        --trainset standard \
        --de_type denoise derain dehaze \
        --epochs 120 \
        --batch_size 8 \
        --lr 2e-4 \
        --num_gpus ${NUM_GPUS} \
        --dim 48 \
        --num_blocks 4 6 6 8 \
        --heads 1 2 4 8 \
        --num_refinement_blocks 4 \
        --gate_type elementwise \
        --LoRa_ffn_ratio 0.5 \
        --LoRa_attn_ratio 0.8 \
        --use_dfrg True \
        --use_ema True \
        --ema_decay 0.999 \
        --loss_type mixed \
        --perceptual_weight 0.1 \
        --edge_weight 0.05 \
        --gradient_clip_val 1.0 \
        --warmup_epochs 15 \
        --patch_size 128 \
        --num_workers 8 \
        --accum_grad 1 \
        --data_file_dir ${DATA_DIR} \
        --ckpt_dir ${CKPT_DIR} \
        --wblogger

    echo "✅ AIO-3 Training Complete!"
}


# ==============================================================================
# Training Script 2: AIO-5 (Five degradations)
# Denoising + Deraining + Dehazing + Deblurring + Low-light Enhancement
# ==============================================================================
train_aio5() {
    echo ""
    echo "🚀 Training AIO-5 Model (All degradations)"
    echo "=================================================="

    python train_improved.py \
        --model model_improved \
        --trainset standard \
        --de_type denoise derain dehaze deblur lol \
        --epochs 120 \
        --batch_size 8 \
        --lr 2e-4 \
        --num_gpus ${NUM_GPUS} \
        --dim 48 \
        --num_blocks 4 6 6 8 \
        --heads 1 2 4 8 \
        --num_refinement_blocks 4 \
        --gate_type elementwise \
        --LoRa_ffn_ratio 0.5 \
        --LoRa_attn_ratio 0.8 \
        --use_dfrg True \
        --use_ema True \
        --ema_decay 0.999 \
        --loss_type mixed \
        --perceptual_weight 0.1 \
        --edge_weight 0.05 \
        --gradient_clip_val 1.0 \
        --warmup_epochs 15 \
        --patch_size 128 \
        --num_workers 8 \
        --accum_grad 1 \
        --data_file_dir ${DATA_DIR} \
        --ckpt_dir ${CKPT_DIR} \
        --wblogger

    echo "✅ AIO-5 Training Complete!"
}


# ==============================================================================
# Training Script 3: CDD11 - Composite Degradations
# Train for 200 epochs as per paper protocol
# ==============================================================================
train_cdd11_all() {
    echo ""
    echo "🚀 Training CDD11 Model (Composite Degradations)"
    echo "=================================================="

    python train_improved.py \
        --model model_improved \
        --trainset CDD11_all \
        --epochs 200 \
        --batch_size 6 \
        --lr 2e-4 \
        --num_gpus ${NUM_GPUS} \
        --dim 48 \
        --num_blocks 4 6 6 8 \
        --heads 1 2 4 8 \
        --num_refinement_blocks 4 \
        --gate_type elementwise \
        --LoRa_ffn_ratio 0.5 \
        --LoRa_attn_ratio 0.8 \
        --use_dfrg True \
        --use_ema True \
        --ema_decay 0.999 \
        --loss_type mixed \
        --perceptual_weight 0.1 \
        --edge_weight 0.05 \
        --gradient_clip_val 1.0 \
        --warmup_epochs 15 \
        --patch_size 128 \
        --num_workers 8 \
        --accum_grad 1 \
        --data_file_dir ${DATA_DIR} \
        --ckpt_dir ${CKPT_DIR} \
        --wblogger

    echo "✅ CDD11 Training Complete!"
}


# ==============================================================================
# Training Script 4: Lightweight Model (Reduced parameters)
# For faster training and deployment
# ==============================================================================
train_lightweight() {
    echo ""
    echo "🚀 Training Lightweight Model"
    echo "=================================================="

    python train_improved.py \
        --model model_improved \
        --trainset standard \
        --de_type denoise derain dehaze \
        --epochs 120 \
        --batch_size 12 \
        --lr 2e-4 \
        --num_gpus ${NUM_GPUS} \
        --dim 32 \
        --num_blocks 4 4 4 6 \
        --heads 1 2 4 8 \
        --num_refinement_blocks 2 \
        --gate_type elementwise \
        --LoRa_ffn_ratio 0.4 \
        --LoRa_attn_ratio 0.6 \
        --use_dfrg True \
        --use_ema True \
        --loss_type charbonnier \
        --gradient_clip_val 1.0 \
        --warmup_epochs 10 \
        --patch_size 128 \
        --num_workers 8 \
        --data_file_dir ${DATA_DIR} \
        --ckpt_dir ${CKPT_DIR}

    echo "✅ Lightweight Training Complete!"
}


# ==============================================================================
# Fine-tuning Script
# Fine-tune from a pre-trained checkpoint
# ==============================================================================
finetune_model() {
    PRETRAIN_CKPT=$1

    if [ -z "$PRETRAIN_CKPT" ]; then
        echo "❌ Error: Please provide checkpoint ID"
        echo "Usage: finetune_model <checkpoint_id>"
        return 1
    fi

    echo ""
    echo "🚀 Fine-tuning from checkpoint: ${PRETRAIN_CKPT}"
    echo "=================================================="

    python train_improved.py \
        --model model_improved \
        --trainset standard \
        --de_type denoise derain dehaze deblur lol \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-4 \
        --num_gpus ${NUM_GPUS} \
        --fine_tune_from ${PRETRAIN_CKPT} \
        --use_ema True \
        --loss_type mixed \
        --perceptual_weight 0.1 \
        --edge_weight 0.05 \
        --patch_size 128 \
        --num_workers 8 \
        --data_file_dir ${DATA_DIR} \
        --ckpt_dir ${CKPT_DIR}

    echo "✅ Fine-tuning Complete!"
}


# ==============================================================================
# Resume Training from Checkpoint
# ==============================================================================
resume_training() {
    RESUME_CKPT=$1

    if [ -z "$RESUME_CKPT" ]; then
        echo "❌ Error: Please provide checkpoint ID"
        echo "Usage: resume_training <checkpoint_id>"
        return 1
    fi

    echo ""
    echo "🚀 Resuming training from: ${RESUME_CKPT}"
    echo "=================================================="

    python train_improved.py \
        --model model_improved \
        --resume_from ${RESUME_CKPT} \
        --num_gpus ${NUM_GPUS} \
        --data_file_dir ${DATA_DIR} \
        --ckpt_dir ${CKPT_DIR}

    echo "✅ Training Resumed!"
}


# ==============================================================================
# Main Menu
# ==============================================================================
show_menu() {
    echo ""
    echo "=================================================="
    echo "Select Training Configuration:"
    echo "=================================================="
    echo "1) AIO-3 (Denoise + Derain + Dehaze)"
    echo "2) AIO-5 (All 5 degradations)"
    echo "3) CDD11 (Composite degradations)"
    echo "4) Lightweight Model"
    echo "5) Fine-tune from checkpoint"
    echo "6) Resume training"
    echo "7) Exit"
    echo "=================================================="
    read -p "Enter choice [1-7]: " choice

    case $choice in
        1) train_aio3 ;;
        2) train_aio5 ;;
        3) train_cdd11_all ;;
        4) train_lightweight ;;
        5)
            read -p "Enter checkpoint ID: " ckpt_id
            finetune_model $ckpt_id
            ;;
        6)
            read -p "Enter checkpoint ID: " ckpt_id
            resume_training $ckpt_id
            ;;
        7) echo "Exiting..."; exit 0 ;;
        *) echo "❌ Invalid choice"; show_menu ;;
    esac
}


# ==============================================================================
# Entry Point
# ==============================================================================
if [ "$1" == "aio3" ]; then
    train_aio3
elif [ "$1" == "aio5" ]; then
    train_aio5
elif [ "$1" == "cdd11" ]; then
    train_cdd11_all
elif [ "$1" == "lightweight" ]; then
    train_lightweight
elif [ "$1" == "finetune" ]; then
    finetune_model $2
elif [ "$1" == "resume" ]; then
    resume_training $2
else
    show_menu
fi
