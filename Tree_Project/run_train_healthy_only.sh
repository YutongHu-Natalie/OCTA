#!/bin/bash

# Training script for DRAC classifier - HEALTHY ONLY (anomaly detection style)
# This is for experimental comparison - NOT RECOMMENDED for your use case

echo "=========================================="
echo "Training DRAC Classifier (Healthy Only)"
echo "=========================================="
echo ""
echo "WARNING: This trains only on healthy samples."
echo "This is for experimental comparison only."
echo "For your explainability goals, use run_train_classifier.sh instead."
echo ""

# Configuration - MODIFY THESE PATHS
DRAC_ROOT="/path/to/DRAC"  # UPDATE THIS to your DRAC dataset path
SAVE_DIR="./results/drac_classifier_healthy_only"
DEVICE="cuda"  # Use "cpu" if no GPU available

# Model hyperparameters
BATCH_SIZE=16
HIDDEN_CHANNELS=128
LATENT_DIM=64
NUM_INTERACTIONS=3
CUTOFF=10.0
READOUT="add"

# Training parameters
EPOCHS=200
LR=0.001
WEIGHT_DECAY=0.0005
SPLIT="811"
TEST_EVERY=5

echo "Configuration:"
echo "  DRAC Root: $DRAC_ROOT"
echo "  Save Directory: $SAVE_DIR"
echo "  Training on: HEALTHY SAMPLES ONLY"
echo "  Device: $DEVICE"
echo ""
echo "Starting training..."
echo ""

python train_drac_classifier.py \
    --drac_root "$DRAC_ROOT" \
    --split "$SPLIT" \
    --batch_size $BATCH_SIZE \
    --hidden_channels $HIDDEN_CHANNELS \
    --latent_dim $LATENT_DIM \
    --num_interactions $NUM_INTERACTIONS \
    --cutoff $CUTOFF \
    --readout "$READOUT" \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device "$DEVICE" \
    --save_dir "$SAVE_DIR" \
    --test_every $TEST_EVERY \
    --healthy_only

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved to: $SAVE_DIR"
echo ""
