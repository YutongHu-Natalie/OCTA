#!/bin/bash

# Complete Two-Stage Training Pipeline
# Stage 1: Pretrain with reconstruction
# Stage 2: Fine-tune for classification

echo "=========================================="
echo "Two-Stage SGMP Training Pipeline"
echo "=========================================="
echo ""
echo "This implements the approach mentioned by the professor:"
echo "  Stage 1: Pretrain SGMP with reconstruction loss"
echo "  Stage 2: Fine-tune encoder for classification"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

# Stage 1: Pretrain
echo ""
echo "=========================================="
echo "Starting Stage 1: Pretraining"
echo "=========================================="
./run_stage1_pretrain.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 1 (pretraining) failed!"
    exit 1
fi

# Stage 2: Fine-tune
echo ""
echo "=========================================="
echo "Starting Stage 2: Fine-tuning"
echo "=========================================="
./run_stage2_finetune.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 2 (fine-tuning) failed!"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "Two-Stage Training Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Stage 1 (Pretrain): ./results/stage1_pretrain/"
echo "  Stage 2 (Finetune): ./results/stage2_finetune/"
echo ""
echo "Final model: ./results/stage2_finetune/best_finetuned_model.pt"
echo "Final results: ./results/stage2_finetune/results.txt"
echo ""
