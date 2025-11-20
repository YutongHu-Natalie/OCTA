#!/bin/bash

# STAGE 2: Fine-tune pretrained SGMP for Classification

echo "=========================================="
echo "STAGE 2: Fine-tuning for Classification"
echo "=========================================="
echo ""
echo "This stage loads the pretrained encoder and"
echo "fine-tunes it for healthy vs unhealthy classification"
echo ""

# Configuration
DRAC_ROOT="../DRAC"
PRETRAINED_MODEL="./results/stage1_pretrain/pretrained_autoencoder.pt"
SAVE_DIR="./results/stage2_finetune"
DEVICE="cuda"

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "ERROR: Pretrained model not found at: $PRETRAINED_MODEL"
    echo ""
    echo "Please run Stage 1 first:"
    echo "  ./run_stage1_pretrain.sh"
    echo ""
    exit 1
fi

# Training parameters
BATCH_SIZE=16
EPOCHS=100
LR=0.0001              # Lower LR for fine-tuning
WEIGHT_DECAY=0.0005
SPLIT="811"
TEST_EVERY=5

# Fine-tuning strategy
FREEZE_ENCODER=false    # Set to true to only train classification head

echo "Configuration:"
echo "  Pretrained Model: $PRETRAINED_MODEL"
echo "  Save Directory: $SAVE_DIR"
echo "  Device: $DEVICE"
echo "  Freeze Encoder: $FREEZE_ENCODER"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR (lower than pretrain)"
echo ""
echo "Starting Stage 2: Fine-tuning..."
echo ""

# Build command
CMD="python finetune_sgmp_classifier.py \
    --drac_root \"$DRAC_ROOT\" \
    --split \"$SPLIT\" \
    --batch_size $BATCH_SIZE \
    --pretrained_model \"$PRETRAINED_MODEL\" \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device \"$DEVICE\" \
    --save_dir \"$SAVE_DIR\" \
    --test_every $TEST_EVERY"

# Add freeze flag if needed
if [ "$FREEZE_ENCODER" = true ]; then
    CMD="$CMD --freeze_encoder"
    echo "Note: Encoder weights are FROZEN (only training classifier head)"
    echo ""
fi

# Execute
eval $CMD

echo ""
echo "=========================================="
echo "Stage 2 Complete!"
echo "=========================================="
echo "Fine-tuned model saved to:"
echo "  $SAVE_DIR/best_finetuned_model.pt"
echo ""
echo "Check results in:"
echo "  $SAVE_DIR/results.txt"
echo ""
