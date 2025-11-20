#!/bin/bash

# Training script for DRAC using SGMP directly (matching original implementation)
# This follows the same approach as main_base_2.py

echo "=========================================="
echo "Training DRAC with SGMP (Original Style)"
echo "=========================================="
echo ""

# Configuration - MODIFY THESE PATHS
DRAC_ROOT="../DRAC"  # UPDATE THIS to your DRAC dataset path
SAVE_DIR="./results/drac_sgmp"
DEVICE="cuda"  # Use "cpu" if no GPU available

# Model hyperparameters (matching original SGMP usage)
BATCH_SIZE=16
HIDDEN_CHANNELS=64      # Original uses 64
NUM_INTERACTIONS=3      # Number of SGMP layers
CUTOFF=10.0
READOUT="add"

# Training parameters
EPOCHS=200
LR=0.001
WEIGHT_DECAY=0.0005
SPLIT="811"  # 80% train, 10% valid, 10% test
TEST_EVERY=5

echo "Configuration:"
echo "  DRAC Root: $DRAC_ROOT"
echo "  Save Directory: $SAVE_DIR"
echo "  Device: $DEVICE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Hidden Channels: $HIDDEN_CHANNELS"
echo "  Num Interactions: $NUM_INTERACTIONS"
echo "  Epochs: $EPOCHS"
echo ""
echo "This script uses SGMP directly as in main_base_2.py"
echo "No separate encoder/decoder - SGMP has built-in classification head"
echo ""
echo "Starting training..."
echo ""

python train_drac_sgmp.py \
    --drac_root "$DRAC_ROOT" \
    --split "$SPLIT" \
    --batch_size $BATCH_SIZE \
    --hidden_channels $HIDDEN_CHANNELS \
    --num_interactions $NUM_INTERACTIONS \
    --cutoff $CUTOFF \
    --readout "$READOUT" \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device "$DEVICE" \
    --save_dir "$SAVE_DIR" \
    --test_every $TEST_EVERY

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved to: $SAVE_DIR"
echo ""
echo "To extract latent representations for analysis:"
echo "  Use the hidden layer output before the final classification layers"
echo ""
