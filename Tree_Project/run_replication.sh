#!/bin/bash

# GT-SSL Replication Run - January 15, 2026
# Organized results directory to compare with original paper
# Run this on remote machine with CUDA support

set -e  # Exit on any error

RESULTS_DIR="./results/replication_2026_01_15"
PRETRAIN_DIR="$RESULTS_DIR/pretrain"
FINETUNE_DIR="$RESULTS_DIR/finetune"
LOG_FILE="$RESULTS_DIR/replication_run.log"

# Create results directories
mkdir -p "$PRETRAIN_DIR"
mkdir -p "$FINETUNE_DIR"

# Function to log to both console and file
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "GT-SSL Replication Run"
log "Started: $(date)"
log "Results will be saved to: $RESULTS_DIR"
log "=========================================="
log ""

# Configuration
DRAC_ROOT="../DRAC"
DEVICE="cuda"  # Use CUDA on remote machine

# Model hyperparameters (same as original)
BATCH_SIZE=4
HIDDEN_CHANNELS=64
NUM_INTERACTIONS=3
CUTOFF=10.0
READOUT="add"
NUM_RBF_CENTERS=20

# GT-SSL hyperparameters
DELTA_MARGIN=1.0
LAMBDA_ORDER=1.0

# Training parameters
PRETRAIN_EPOCHS=100
FINETUNE_EPOCHS=100
PRETRAIN_LR=0.001
FINETUNE_LR=0.0001
WEIGHT_DECAY=0.0005
SPLIT="811"
TEST_EVERY=5

# Log configuration
log "Configuration:"
log "  DRAC_ROOT: $DRAC_ROOT"
log "  DEVICE: $DEVICE"
log "  BATCH_SIZE: $BATCH_SIZE"
log "  HIDDEN_CHANNELS: $HIDDEN_CHANNELS"
log "  PRETRAIN_EPOCHS: $PRETRAIN_EPOCHS"
log "  FINETUNE_EPOCHS: $FINETUNE_EPOCHS"
log ""

# ==========================================
# STAGE 1: GT-SSL Pretraining
# ==========================================
log ""
log "=========================================="
log "STAGE 1: GT-SSL Pretraining"
log "=========================================="
log "Device: $DEVICE"
log "Epochs: $PRETRAIN_EPOCHS"
log "Learning Rate: $PRETRAIN_LR"
log "Save to: $PRETRAIN_DIR"
log ""

python pretrain_gtssl.py \
    --drac_root "$DRAC_ROOT" \
    --split "$SPLIT" \
    --batch_size $BATCH_SIZE \
    --hidden_channels $HIDDEN_CHANNELS \
    --num_interactions $NUM_INTERACTIONS \
    --cutoff $CUTOFF \
    --readout "$READOUT" \
    --num_rbf_centers $NUM_RBF_CENTERS \
    --delta_margin $DELTA_MARGIN \
    --lambda_order $LAMBDA_ORDER \
    --epochs $PRETRAIN_EPOCHS \
    --lr $PRETRAIN_LR \
    --weight_decay $WEIGHT_DECAY \
    --device "$DEVICE" \
    --save_dir "$PRETRAIN_DIR" \
    --test_every $TEST_EVERY \
    2>&1 | tee -a "$LOG_FILE"

log ""
log "Stage 1 Complete! Model saved to: $PRETRAIN_DIR/best_gtssl_model.pt"
log ""

# ==========================================
# STAGE 2: Fine-tuning for Classification
# ==========================================
log ""
log "=========================================="
log "STAGE 2: Fine-tuning for Classification"
log "=========================================="
log "Pretrained model: $PRETRAIN_DIR/best_gtssl_model.pt"
log "Device: $DEVICE"
log "Epochs: $FINETUNE_EPOCHS"
log "Learning Rate: $FINETUNE_LR"
log "Save to: $FINETUNE_DIR"
log ""

python finetune_sgmp_classifier.py \
    --drac_root "$DRAC_ROOT" \
    --split "$SPLIT" \
    --batch_size 16 \
    --pretrained_model "$PRETRAIN_DIR/best_gtssl_model.pt" \
    --epochs $FINETUNE_EPOCHS \
    --lr $FINETUNE_LR \
    --weight_decay $WEIGHT_DECAY \
    --device "$DEVICE" \
    --save_dir "$FINETUNE_DIR" \
    --test_every $TEST_EVERY \
    2>&1 | tee -a "$LOG_FILE"

# ==========================================
# Summary
# ==========================================
log ""
log "=========================================="
log "GT-SSL Replication Complete!"
log "Finished: $(date)"
log "=========================================="
log ""
log "Results saved to:"
log "  Pretrain: $PRETRAIN_DIR/"
log "  Finetune: $FINETUNE_DIR/"
log ""
log "Key output files:"
log "  Pretrained model: $PRETRAIN_DIR/best_gtssl_model.pt"
log "  Final model: $FINETUNE_DIR/best_finetuned_model.pt"
log "  Results: $FINETUNE_DIR/results.txt"
log "  Full log: $LOG_FILE"
log ""
