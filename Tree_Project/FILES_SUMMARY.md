# Files Summary

This document lists all files created for the DRAC classifier project.

## 📋 Documentation Files

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | **START HERE** - Quick setup and usage guide |
| `README_DRAC_CLASSIFIER.md` | Comprehensive documentation with theory and examples |
| `FILES_SUMMARY.md` | This file - overview of all files |
| `requirements.txt` | Python dependencies list |

## 🚀 Shell Scripts (Executable)

| File | Command | Purpose |
|------|---------|---------|
| `setup_environment.sh` | `./setup_environment.sh` | Check dependencies and make scripts executable |
| `run_train_classifier.sh` | `./run_train_classifier.sh` | Train classifier on both healthy + unhealthy (**recommended**) |
| `run_train_healthy_only.sh` | `./run_train_healthy_only.sh` | Train on healthy only (experimental comparison) |
| `run_analyze_latents.sh` | `./run_analyze_latents.sh` | Analyze latent space after training |
| `run_example_workflow.sh` | `./run_example_workflow.sh` | Run example demonstrations |
| `run_full_pipeline.sh` | `./run_full_pipeline.sh` | Complete pipeline: train → analyze → examples |

## 🧠 Model Files

| File | Purpose |
|------|---------|
| `models/graph_autoencoder.py` | Core model implementations:<br>- `GraphEncoder`: SGMP-based encoder<br>- `GraphDecoder`: Decoder for reconstruction<br>- `GraphAutoencoder`: Full autoencoder<br>- `GraphClassifier`: Binary classifier |
| `models/SGMP.py` | Self-supervised Geometric Message Passing (existing) |

## 📊 Data Loading

| File | Purpose |
|------|---------|
| `drac_data_loader.py` | Enhanced DRAC dataset loader with:<br>- Binary classification support<br>- Label filtering<br>- Automatic distribution reporting |

## 🏃 Training Scripts

| File | Purpose |
|------|---------|
| `train_drac_classifier.py` | Main training script for binary classifier<br>Supports:<br>- Training on both classes or healthy only<br>- Pretrained encoder loading<br>- Automatic latent extraction |

## 📈 Analysis Tools

| File | Purpose |
|------|---------|
| `latent_analysis.py` | Comprehensive latent space analysis:<br>- Visualize with PCA/t-SNE<br>- Identify discriminative dimensions<br>- Perturb latent vectors<br>- Interpolate between samples<br>- Find nearest neighbors<br>- Compute healthy→unhealthy direction |

## 💡 Examples

| File | Purpose |
|------|---------|
| `example_workflow.py` | Demonstration of complete workflow:<br>- Model inference<br>- Latent analysis<br>- Perturbations<br>- Interpolation<br>- Nearest neighbors |

## 📂 Directory Structure After Running

```
Tree_Project/
│
├── 📄 Documentation
│   ├── QUICKSTART.md                    ← Start here!
│   ├── README_DRAC_CLASSIFIER.md        ← Full docs
│   ├── FILES_SUMMARY.md                 ← This file
│   └── requirements.txt                 ← Dependencies
│
├── 🚀 Shell Scripts
│   ├── setup_environment.sh
│   ├── run_train_classifier.sh          ← Main training script
│   ├── run_train_healthy_only.sh
│   ├── run_analyze_latents.sh
│   ├── run_example_workflow.sh
│   └── run_full_pipeline.sh
│
├── 🧠 Models
│   └── models/
│       ├── graph_autoencoder.py         ← New models
│       └── SGMP.py                      ← Existing encoder
│
├── 📊 Data & Training
│   ├── drac_data_loader.py              ← Modified loader
│   ├── train_drac_classifier.py         ← Training script
│   ├── latent_analysis.py               ← Analysis tools
│   └── example_workflow.py              ← Examples
│
└── 📁 Results (created after running)
    └── results/
        ├── drac_classifier/
        │   ├── best_model.pt            ← Trained model
        │   ├── results.txt              ← Metrics
        │   └── latent_representations.npz  ← Latents
        └── analysis/
            ├── latent_space_pca.png     ← Visualizations
            ├── latent_space_tsne.png
            ├── discriminative_dims.png
            └── latent_statistics.png
```

## 🎯 Quick Reference

### First Time Setup
```bash
./setup_environment.sh                  # Check dependencies
pip install -r requirements.txt         # Install dependencies
# Edit run_train_classifier.sh to set DRAC_ROOT path
```

### Training
```bash
./run_train_classifier.sh              # Train (recommended)
# OR
./run_full_pipeline.sh                 # Complete pipeline
```

### Analysis
```bash
./run_analyze_latents.sh               # After training
```

### Examples
```bash
./run_example_workflow.sh              # Demonstrations
```

## 📝 File Sizes (Approximate)

- Shell scripts: ~1-3 KB each
- Python scripts: ~5-15 KB each
- Documentation: ~5-20 KB each
- Trained model: ~10-50 MB (depends on hyperparameters)
- Latent representations: ~1-10 MB (depends on dataset size)

## 🔄 Workflow Summary

```
┌─────────────────┐
│  Setup          │
│  (one-time)     │
└────────┬────────┘
         │
         ├─→ ./setup_environment.sh
         └─→ Edit DRAC_ROOT in scripts

┌─────────────────┐
│  Training       │
└────────┬────────┘
         │
         ├─→ ./run_train_classifier.sh
         │   (or ./run_full_pipeline.sh)
         └─→ Results in ./results/drac_classifier/

┌─────────────────┐
│  Analysis       │
└────────┬────────┘
         │
         ├─→ ./run_analyze_latents.sh
         └─→ Visualizations in ./results/analysis/

┌─────────────────┐
│  Experiments    │
└────────┬────────┘
         │
         ├─→ ./run_example_workflow.sh
         ├─→ Modify latent_analysis.py for custom analysis
         └─→ Use discriminative dims for perturbations
```

## 🆘 Getting Help

1. **Quick start**: Read `QUICKSTART.md`
2. **Detailed info**: Check `README_DRAC_CLASSIFIER.md`
3. **Code examples**: See `example_workflow.py`
4. **Troubleshooting**: Section in `QUICKSTART.md`

## ✅ Checklist

Before starting:
- [ ] Read `QUICKSTART.md`
- [ ] Run `./setup_environment.sh`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Update `DRAC_ROOT` path in `run_train_classifier.sh`
- [ ] Have DRAC dataset ready

Ready to train:
- [ ] Run `./run_train_classifier.sh`
- [ ] Wait for training to complete (~1-2 hours)
- [ ] Check `results/drac_classifier/results.txt`

Ready to analyze:
- [ ] Run `./run_analyze_latents.sh`
- [ ] Review visualizations in `results/analysis/`
- [ ] Note discriminative dimensions from output

Ready for experiments:
- [ ] Run `./run_example_workflow.sh`
- [ ] Modify `latent_analysis.py` for custom perturbations
- [ ] Implement graph decoder for visualization (future)
