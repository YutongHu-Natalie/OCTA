#!/bin/bash

# Script to run example workflow demonstrating the complete pipeline

echo "=========================================="
echo "Running Example Workflow"
echo "=========================================="
echo ""

# Check if trained model exists
MODEL_PATH="./results/drac_classifier/best_model.pt"
LATENTS_PATH="./results/drac_classifier/latent_representations.npz"

if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Trained model not found at: $MODEL_PATH"
    echo ""
    echo "Some examples may fail. Please train a model first using:"
    echo "  ./run_train_classifier.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Running examples..."
echo ""

python example_workflow.py

echo ""
echo "=========================================="
echo "Examples Complete!"
echo "=========================================="
echo ""
