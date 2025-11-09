# DRAC OCTA Graph Classifier with Latent Space Analysis

This repository contains a graph-based classifier for DRAC (Diabetic Retinopathy Analysis Challenge) OCTA images using the SGMP (Self-supervised Geometric Message Passing) encoder.

## Overview

The system converts OCTA retinal images into geometric graphs and learns discriminative latent representations to classify healthy vs unhealthy samples. The learned latent space can be analyzed and perturbed to understand disease-related features.

## Architecture

1. **Graph Encoder (SGMP)**: Self-supervised geometric encoder that processes 3D graph structures
2. **Graph Classifier**: Binary classification head for healthy/unhealthy prediction
3. **Latent Analysis Tools**: Utilities for understanding and perturbing the latent space

## Why Train on Both Healthy AND Unhealthy Data?

**Recommendation: Train on BOTH classes, not just healthy samples.**

### Reasons:

1. **Balanced Dataset**: You have ~50/50 healthy/unhealthy distribution, perfect for supervised learning
2. **Better Latent Structure**: Training with both classes creates a latent space where:
   - Classes cluster separately
   - Boundaries represent clinically meaningful transitions
   - Perturbations along discriminative axes show disease progression
3. **Interpretability**: You can identify which latent dimensions encode disease features
4. **Classification Performance**: Direct supervision gives much better results than anomaly detection
5. **Clinical Insights**: Enables counterfactual analysis ("what would make this sample healthy?")

Anomaly detection (training only on healthy) is useful when:
- You have very few unhealthy samples
- You want to detect novel pathologies not seen during training
- This is NOT your case!

## Files Created

### Models
- `models/graph_autoencoder.py`: Contains:
  - `GraphEncoder`: SGMP-based encoder to latent space
  - `GraphDecoder`: Decoder for reconstruction (optional, for future autoencoder pretraining)
  - `GraphAutoencoder`: Complete autoencoder (for unsupervised pretraining if desired)
  - `GraphClassifier`: Classification model with SGMP encoder

### Data Loading
- `drac_data_loader.py` (modified): Enhanced with:
  - `filter_labels`: Filter to specific labels (e.g., healthy only for experiments)
  - `binary_classification`: Convert multi-class to binary (healthy vs unhealthy)
  - Automatic label distribution reporting

### Training
- `train_drac_classifier.py`: Main training script for binary classifier

### Analysis
- `latent_analysis.py`: Comprehensive latent space analysis tools

## Usage

### 1. Train the Classifier

Train on both healthy and unhealthy samples (recommended):

```bash
python train_drac_classifier.py \
    --drac_root /path/to/DRAC/dataset \
    --split 811 \
    --batch_size 16 \
    --hidden_channels 128 \
    --latent_dim 64 \
    --num_interactions 3 \
    --epochs 200 \
    --lr 1e-3 \
    --device cuda \
    --save_dir ./results/drac_classifier
```

For experimental comparison, train only on healthy samples (anomaly detection style):

```bash
python train_drac_classifier.py \
    --drac_root /path/to/DRAC/dataset \
    --healthy_only \
    --save_dir ./results/drac_classifier_healthy_only \
    [other args...]
```

### 2. Analyze Latent Space

After training, analyze the learned latent representations:

```bash
python latent_analysis.py \
    --latents_file ./results/drac_classifier/latent_representations.npz \
    --output_dir ./results/analysis
```

This will:
1. Visualize latent space with PCA and t-SNE
2. Identify discriminative dimensions (which dimensions separate healthy/unhealthy)
3. Plot distributions of top discriminative dimensions
4. Show latent space statistics
5. Compute the healthy→unhealthy direction vector

### 3. Use the Trained Model

Load and use the trained classifier:

```python
import torch
from models.graph_autoencoder import GraphClassifier

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphClassifier(
    input_channels_node=5,
    hidden_channels=128,
    latent_dim=64,
    num_classes=2
).to(device)

checkpoint = torch.load('./results/drac_classifier/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions with latent extraction
with torch.no_grad():
    logits, latent = model(x, pos, batch, edge_index_3rd, return_latent=True)
    prediction = logits.argmax(dim=1)  # 0 = healthy, 1 = unhealthy
```

### 4. Perturb Latent Space

Use the analysis tools to perturb latent representations:

```python
from latent_analysis import LatentSpaceAnalyzer

# Load analyzer
analyzer = LatentSpaceAnalyzer('./results/drac_classifier/latent_representations.npz')

# Find discriminative dimensions
results = analyzer.identify_discriminative_dimensions(top_k=10)
most_discriminative_dim = results[0]['dimension']

# Perturb along discriminative dimension
base_latent = analyzer.test_latents[0]  # Take a test sample
perturbations = analyzer.create_perturbation_vectors(
    dimension=most_discriminative_dim,
    num_steps=10,
    scale=3.0
)

# Create perturbed versions
perturbed_latents = []
for p in perturbations:
    perturbed = analyzer.perturb_latent(base_latent, most_discriminative_dim, p)
    perturbed_latents.append(perturbed)

# Decode with model to visualize changes (future work)
# This requires implementing graph decoding or visualization
```

### 5. Interpolation Between Samples

Interpolate between a healthy and unhealthy sample:

```python
# Find a healthy and unhealthy sample
healthy_idx = np.where(analyzer.test_labels == 0)[0][0]
unhealthy_idx = np.where(analyzer.test_labels == 1)[0][0]

healthy_latent = analyzer.test_latents[healthy_idx]
unhealthy_latent = analyzer.test_latents[unhealthy_idx]

# Create interpolation path
interpolated = analyzer.interpolate_between_samples(
    healthy_latent,
    unhealthy_latent,
    num_steps=10
)

# Pass through classifier to see how predictions change
for z in interpolated:
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)
    # Note: This requires modifying the model to accept latent vectors directly
```

## Model Parameters

### Important Hyperparameters:

- `hidden_channels`: Hidden dimension size (default: 128)
- `latent_dim`: Latent space dimension (default: 64)
  - Larger = more expressive but harder to interpret
  - Smaller = more compact, potentially more interpretable
- `num_interactions`: Number of SGMP message passing layers (default: 3)
- `cutoff`: Distance cutoff for geometric features (default: 10.0)
- `readout`: Graph pooling method ('add', 'mean', or 'sum')

### Training Parameters:

- `epochs`: Number of training epochs (default: 200)
- `lr`: Learning rate (default: 1e-3)
- `batch_size`: Batch size (default: 16)
- `weight_decay`: L2 regularization (default: 5e-4)

## Expected Results

After training, you should see:

- **Accuracy**: 70-90% on test set (depends on data quality and balance)
- **ROC-AUC**: 0.80-0.95 for binary classification
- **Latent Space**: Clear separation between healthy and unhealthy clusters
- **Discriminative Dimensions**: 5-10 dimensions with large effect sizes (Cohen's d > 0.5)

## Future Work: Decoder Implementation

To fully visualize latent perturbations, you can:

1. Train the autoencoder with reconstruction loss:
   - Implement graph reconstruction from `GraphDecoder`
   - Train to minimize reconstruction error
   - Fine-tune classifier on top

2. Use decoder to visualize perturbations:
   - Perturb latent vector
   - Decode to graph structure
   - Visualize changes in graph topology or node features

3. Implement counterfactual generation:
   - Find minimal changes to make unhealthy → healthy
   - Identify disease-critical graph features

## Data Requirements

Your DRAC dataset should be organized as:
```
DRAC/
├── C. Diabetic Retinopathy Grading/
│   ├── 1. Original Images/
│   │   └── a. Training Set/
│   │       ├── image1.png
│   │       ├── image2.png
│   │       └── ...
│   └── 2. Groundtruths/
│       └── a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv
```

The CSV should contain:
- `image name`: Filename of the image
- `DR grade`: Label (0 = healthy, 1+ = unhealthy grades)

## Dependencies

```bash
torch
torch-geometric
torch-scatter
numpy
pandas
scikit-learn
scikit-image
scipy
matplotlib
seaborn
```

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size`
- Reduce `hidden_channels` or `latent_dim`
- Use smaller images (modify transform in data loader)

### Poor classification performance
- Try longer training (`--epochs 300`)
- Adjust learning rate (`--lr 5e-4`)
- Check class balance in data loader output
- Ensure data quality (check image preprocessing)
e
### Latent space not separating
- Increase `latent_dim` for more capacity
- Add more `num_interactions` layers
- Try different `readout` methods
- Check if encoder is learning (monitor training loss)

