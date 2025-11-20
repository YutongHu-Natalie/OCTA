#!/bin/bash

# Complete GT-SSL Two-Stage Training Pipeline
# Stage 1: GT-SSL pretraining (partial ordering + subtree growth)
# Stage 2: Fine-tune for classification

echo "=========================================="
echo "GT-SSL Two-Stage Training Pipeline"
echo "=========================================="
echo ""
echo "This implements the CORRECT approach from the paper:"
echo "  Stage 1: GT-SSL pretraining"
echo "    - Partial Ordering Constraint (hierarchy)"
echo "    - Subtree Growth Learning (geometry)"
echo "  Stage 2: Fine-tune for classification"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

# Stage 1: GT-SSL Pretrain
echo ""
echo "=========================================="
echo "Starting Stage 1: GT-SSL Pretraining"
echo "=========================================="
./run_gtssl_pretrain.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 1 (GT-SSL pretraining) failed!"
    exit 1
fi

# Stage 2: Fine-tune
echo ""
echo "=========================================="
echo "Starting Stage 2: Fine-tuning"
echo "=========================================="
./run_gtssl_finetune.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 2 (fine-tuning) failed!"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "GT-SSL Two-Stage Training Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Stage 1 (GT-SSL): ./results/gtssl_pretrain/"
echo "  Stage 2 (Finetune): ./results/gtssl_finetune/"
echo ""
echo "Final model: ./results/gtssl_finetune/best_finetuned_model.pt"
echo "Final results: ./results/gtssl_finetune/results.txt"
echo ""
