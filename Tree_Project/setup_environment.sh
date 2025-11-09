#!/bin/bash

# Setup script: Make all shell scripts executable and check dependencies

echo "=========================================="
echo "Setting Up DRAC Classifier Environment"
echo "=========================================="
echo ""

# Make all shell scripts executable
echo "Making shell scripts executable..."
chmod +x run_train_classifier.sh
chmod +x run_train_healthy_only.sh
chmod +x run_analyze_latents.sh
chmod +x run_example_workflow.sh
chmod +x run_full_pipeline.sh
echo "✓ Shell scripts are now executable"
echo ""

# Check Python
echo "Checking Python installation..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "✗ Python not found. Please install Python 3.7+"
    exit 1
fi
echo ""

# Check key dependencies
echo "Checking Python dependencies..."

dependencies=(
    "torch"
    "torch_geometric"
    "torch_scatter"
    "numpy"
    "pandas"
    "sklearn"
    "scipy"
    "matplotlib"
    "seaborn"
    "PIL"
    "skimage"
)

missing_deps=()

for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        echo "✓ $dep"
    else
        echo "✗ $dep (missing)"
        missing_deps+=("$dep")
    fi
done

echo ""

if [ ${#missing_deps[@]} -ne 0 ]; then
    echo "=========================================="
    echo "Missing Dependencies"
    echo "=========================================="
    echo ""
    echo "The following packages are missing:"
    for dep in "${missing_deps[@]}"; do
        echo "  - $dep"
    done
    echo ""
    echo "To install, run:"
    echo "  pip install torch torchvision"
    echo "  pip install torch-geometric torch-scatter"
    echo "  pip install numpy pandas scikit-learn scipy matplotlib seaborn Pillow scikit-image"
    echo ""
    echo "Or use the provided requirements file:"
    echo "  pip install -r requirements.txt"
    echo ""
else
    echo "=========================================="
    echo "All Dependencies Installed!"
    echo "=========================================="
    echo ""
fi

# Check for CUDA
echo "Checking CUDA availability..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    echo "✓ CUDA available (version: $CUDA_VERSION)"
    echo "  You can use --device cuda in training scripts"
else
    echo "⚠ CUDA not available"
    echo "  Training will use CPU (slower). Use --device cpu in scripts."
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Update DRAC_ROOT path in run_train_classifier.sh"
echo "  2. Run: ./run_train_classifier.sh"
echo "  3. Or run the full pipeline: ./run_full_pipeline.sh"
echo ""
echo "Quick reference:"
echo "  ./run_train_classifier.sh      - Train classifier (both classes)"
echo "  ./run_train_healthy_only.sh    - Train on healthy only (experimental)"
echo "  ./run_analyze_latents.sh       - Analyze latent space"
echo "  ./run_example_workflow.sh      - Run examples"
echo "  ./run_full_pipeline.sh         - Complete pipeline"
echo ""
