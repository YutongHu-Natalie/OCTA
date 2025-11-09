# Quick Start Guide

## Setup (One-time)

1. **Setup environment and check dependencies:**
   ```bash
   ./setup_environment.sh
   ```
   This will:
   - Make all scripts executable
   - Check Python installation
   - Verify dependencies
   - Check CUDA availability

2. **Install missing dependencies (if any):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Update DRAC dataset path:**
   Edit `run_train_classifier.sh` and change:
   ```bash
   DRAC_ROOT="/path/to/DRAC"  # Line 9
   ```
   to your actual DRAC dataset path, e.g.:
   ```bash
   DRAC_ROOT="/Users/nataliehu/Desktop/emory/DRAC"
   ```

## Usage

### Option 1: Full Pipeline (Recommended for first run)

Run everything in one go:
```bash
./run_full_pipeline.sh
```
This will:
1. Train the classifier (both healthy + unhealthy)
2. Analyze latent space
3. Run example workflows

**Time:** ~1-3 hours depending on dataset size and GPU

---

### Option 2: Step-by-Step (Recommended)

**Step 1: Train the classifier**
```bash
./run_train_classifier.sh
```
- Trains on both healthy AND unhealthy samples (recommended)
- Saves model to `./results/drac_classifier/`
- Time: ~1-2 hours

**Step 2: Analyze latent space**
```bash
./run_analyze_latents.sh
```
- Generates visualizations and identifies discriminative dimensions
- Saves to `./results/analysis/`
- Time: ~1-5 minutes

**Step 3: Run examples**
```bash
./run_example_workflow.sh
```
- Demonstrates inference, perturbation, interpolation
- Time: ~1 minute

---

### Option 3: Train on Healthy Only (Experimental Comparison)

```bash
./run_train_healthy_only.sh
```
- For comparing anomaly detection vs supervised learning
- NOT recommended for your explainability goals
- Compare results with Option 1 to see the difference

## What You Get

After running, you'll have:

```
Tree_Project/
├── results/
│   ├── drac_classifier/
│   │   ├── best_model.pt                    # Trained model
│   │   ├── results.txt                      # Performance metrics
│   │   └── latent_representations.npz       # All latent vectors
│   └── analysis/
│       ├── latent_space_pca.png             # PCA visualization
│       ├── latent_space_tsne.png            # t-SNE visualization
│       ├── discriminative_dims.png          # Top dimensions
│       └── latent_statistics.png            # Statistical analysis
```

## Results to Check

1. **Performance** (`results/drac_classifier/results.txt`):
   - Test accuracy: Should be 70-90%
   - ROC-AUC: Should be 0.80-0.95
   - Confusion matrix

2. **Latent Space** (`results/analysis/latent_space_*.png`):
   - Check if healthy and unhealthy form separate clusters
   - Good separation = explainable model

3. **Discriminative Dimensions** (terminal output + `discriminative_dims.png`):
   - Which latent dimensions matter for classification
   - Use these for perturbation experiments later

## Customization

### Adjust hyperparameters

Edit `run_train_classifier.sh`:
```bash
BATCH_SIZE=16           # Reduce if out of memory
HIDDEN_CHANNELS=128     # Model capacity
LATENT_DIM=64          # Latent space size
EPOCHS=200             # Training duration
LR=0.001               # Learning rate
```

### Use CPU instead of GPU

Change in `run_train_classifier.sh`:
```bash
DEVICE="cpu"  # Instead of "cuda"
```

### Different data split

Change in `run_train_classifier.sh`:
```bash
SPLIT="811"  # 80% train, 10% val, 10% test
# Or use "721" for 70/20/10, etc.
```

## Troubleshooting

### "DRAC root directory not found"
- Update `DRAC_ROOT` in `run_train_classifier.sh` to your actual path

### "CUDA out of memory"
- Reduce `BATCH_SIZE` (try 8 or 4)
- Or use `DEVICE="cpu"`

### "Module not found"
- Run: `pip install -r requirements.txt`

### Scripts not executable
- Run: `chmod +x *.sh`

## Next Steps After Training

1. **Review discriminative dimensions** from analysis output

2. **Use for perturbations:**
   ```python
   from latent_analysis import LatentSpaceAnalyzer

   analyzer = LatentSpaceAnalyzer('./results/drac_classifier/latent_representations.npz')
   results = analyzer.identify_discriminative_dimensions()

   # Use top dimensions for perturbation experiments
   ```

3. **Implement graph decoder** (future work) to visualize perturbations

4. **Clinical interpretation**: Map discriminative dimensions to medical features

## Questions?

Check:
- Full documentation: `README_DRAC_CLASSIFIER.md`
- Example code: `example_workflow.py`
- Analysis tools: `latent_analysis.py`
