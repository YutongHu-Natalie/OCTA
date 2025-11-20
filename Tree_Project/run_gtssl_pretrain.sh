#!/bin/bash

# STAGE 1: GT-SSL Pretraining (CORRECT APPROACH)
# This implements the actual paper's approach with:
#   - Partial ordering constraint (hierarchical structure)
#   - Subtree growth learning (geometric patterns)
# NO traditional autoencoder reconstruction!

echo "=========================================="
echo "STAGE 1: GT-SSL Pretraining"
echo "=========================================="
echo ""
echo "This implements the CORRECT approach from the paper:"
echo "  - Partial Ordering Constraint (enforce hierarchy)"
echo "  - Subtree Growth Learning (predict child geometry)"
echo "  - NO traditional reconstruction!"
echo ""

# Configuration
DRAC_ROOT="../DRAC"
SAVE_DIR="./results/gtssl_pretrain"
DEVICE="cuda"

# Model hyperparameters
BATCH_SIZE=16
HIDDEN_CHANNELS=64
NUM_INTERACTIONS=3
CUTOFF=10.0
READOUT="add"
NUM_RBF_CENTERS=20

# GT-SSL hyperparameters
DELTA_MARGIN=1.0
LAMBDA_ORDER=1.0

# Training parameters
EPOCHS=100
LR=0.001                # Higher LR for pretrain
WEIGHT_DECAY=0.0005
SPLIT="811"
TEST_EVERY=5

# Data options
TRAIN_ON_HEALTHY_ONLY=""  # Set to "--train_on_healthy_only" to use only healthy samples

echo "Configuration:"
echo "  DRAC Root: $DRAC_ROOT"
echo "  Save Directory: $SAVE_DIR"
echo "  Device: $DEVICE"
echo "  Hidden Channels: $HIDDEN_CHANNELS"
echo "  RBF Centers: $NUM_RBF_CENTERS"
echo "  Delta Margin: $DELTA_MARGIN"
echo "  Lambda Order: $LAMBDA_ORDER"
echo "  Epochs: $EPOCHS"
echo ""
echo "Starting GT-SSL pretraining..."
echo ""

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
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device "$DEVICE" \
    --save_dir "$SAVE_DIR" \
    --test_every $TEST_EVERY \
    $TRAIN_ON_HEALTHY_ONLY

echo ""
echo "=========================================="
echo "GT-SSL Pretraining Complete!"
echo "=========================================="
echo "Model saved to:"
echo "  $SAVE_DIR/best_gtssl_model.pt"
echo ""
echo "Next step: Run Stage 2 (Fine-tuning)"
echo "  ./run_stage2_finetune.sh"
echo ""
