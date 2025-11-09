#!/bin/bash

# Training script for DRAC classifier
# Train on both healthy and unhealthy samples (recommended)

echo "=========================================="
echo "Training DRAC Graph Classifier"
echo "=========================================="
echo ""

# Configuration - MODIFY THESE PATHS
DRAC_ROOT="/path/to/DRAC"  # UPDATE THIS to your DRAC dataset path
SAVE_DIR="./results/drac_classifier"
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
SPLIT="811"  # 80% train, 10% valid, 10% test
TEST_EVERY=5

echo "Configuration:"
echo "  DRAC Root: $DRAC_ROOT"
echo "  Save Directory: $SAVE_DIR"
echo "  Device: $DEVICE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Hidden Channels: $HIDDEN_CHANNELS"
echo "  Latent Dim: $LATENT_DIM"
echo "  Epochs: $EPOCHS"
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
    --test_every $TEST_EVERY

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved to: $SAVE_DIR"
echo ""
echo "Next steps:"
echo "1. Check results in $SAVE_DIR/results.txt"
echo "2. Run analysis: ./run_analyze_latents.sh"
echo ""
