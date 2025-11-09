#!/bin/bash

# Complete pipeline: Train classifier, analyze latents, run examples

echo "=========================================="
echo "DRAC Classifier - Full Pipeline"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Train the graph classifier"
echo "  2. Analyze latent representations"
echo "  3. Run example workflows"
echo ""

# Configuration - MODIFY THIS PATH
DRAC_ROOT="/path/to/DRAC"  # UPDATE THIS to your DRAC dataset path

echo "DRAC Root: $DRAC_ROOT"
echo ""

# Check if DRAC root exists
if [ ! -d "$DRAC_ROOT" ]; then
    echo "ERROR: DRAC root directory not found at: $DRAC_ROOT"
    echo ""
    echo "Please update the DRAC_ROOT variable in this script to point to your DRAC dataset."
    echo ""
    exit 1
fi

read -p "Ready to start? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 1: Training Classifier"
echo "=========================================="
echo ""

# Update DRAC_ROOT in training script and run
sed -i.bak "s|DRAC_ROOT=\".*\"|DRAC_ROOT=\"$DRAC_ROOT\"|" run_train_classifier.sh
./run_train_classifier.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed. Stopping pipeline."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Analyzing Latent Space"
echo "=========================================="
echo ""

./run_analyze_latents.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Analysis failed. Stopping pipeline."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Running Examples"
echo "=========================================="
echo ""

./run_example_workflow.sh

echo ""
echo "=========================================="
echo "Full Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Model: ./results/drac_classifier/best_model.pt"
echo "  - Latents: ./results/drac_classifier/latent_representations.npz"
echo "  - Analysis: ./results/analysis/"
echo ""
echo "Next steps:"
echo "  1. Review results in ./results/drac_classifier/results.txt"
echo "  2. Examine visualizations in ./results/analysis/"
echo "  3. Use discriminative dimensions for perturbation experiments"
echo ""
