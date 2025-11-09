#!/bin/bash

# Script to analyze learned latent representations

echo "=========================================="
echo "Analyzing Latent Space Representations"
echo "=========================================="
echo ""

# Configuration - MODIFY THESE PATHS
LATENTS_FILE="./results/drac_classifier/latent_representations.npz"
OUTPUT_DIR="./results/analysis"

# Check if latent file exists
if [ ! -f "$LATENTS_FILE" ]; then
    echo "ERROR: Latent representations file not found at: $LATENTS_FILE"
    echo ""
    echo "Please train a model first using:"
    echo "  ./run_train_classifier.sh"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  Latents File: $LATENTS_FILE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""
echo "Running analysis..."
echo ""

python latent_analysis.py \
    --latents_file "$LATENTS_FILE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - latent_space_pca.png"
echo "  - latent_space_tsne.png"
echo "  - discriminative_dims.png"
echo "  - latent_statistics.png"
echo ""
echo "Check the terminal output above for:"
echo "  - Top discriminative dimensions"
echo "  - Statistical significance"
echo "  - Healthy→Unhealthy direction"
echo ""
