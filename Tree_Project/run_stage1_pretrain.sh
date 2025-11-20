#!/bin/bash

# STAGE 1: Pretrain SGMP with Autoencoder (Reconstruction)
# This is the MISSING stage mentioned by the professor

echo "=========================================="
echo "STAGE 1: Pretraining SGMP Autoencoder"
echo "=========================================="
echo ""
echo "This stage trains SGMP with reconstruction loss"
echo "to learn good representations before classification"
echo ""

# Configuration
DRAC_ROOT="../DRAC"
SAVE_DIR="./results/stage1_pretrain"
DEVICE="cuda"

# Model hyperparameters
BATCH_SIZE=16
HIDDEN_CHANNELS=64
LATENT_DIM=32           # Compressed latent space
NUM_INTERACTIONS=3
CUTOFF=10.0
READOUT="add"

# Training parameters
EPOCHS=100              # Pretrain for fewer epochs than fine-tune
LR=0.001                # Higher LR for pretrain
WEIGHT_DECAY=0.0005
SPLIT="811"
TEST_EVERY=5

echo "Configuration:"
echo "  DRAC Root: $DRAC_ROOT"
echo "  Save Directory: $SAVE_DIR"
echo "  Device: $DEVICE"
echo "  Latent Dimension: $LATENT_DIM"
echo "  Epochs: $EPOCHS"
echo ""
echo "Starting Stage 1: Pretraining with Reconstruction..."
echo ""

python pretrain_sgmp_autoencoder.py \
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
echo "Stage 1 Complete!"
echo "=========================================="
echo "Pretrained model saved to:"
echo "  $SAVE_DIR/pretrained_autoencoder.pt"
echo ""
echo "Next step: Run Stage 2 (Fine-tuning)"
echo "  ./run_stage2_finetune.sh"
echo ""
