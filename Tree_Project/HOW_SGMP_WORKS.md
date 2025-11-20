# Understanding the Existing SGMP Implementation

## Overview

The existing codebase uses **SGMP (Self-supervised Geometric Message Passing)** as a complete end-to-end model for graph classification. This document explains how it works so you can ask informed questions.

---

## Architecture Flow

```
Input: Graph (x, pos, edge_index)
  ↓
[1] Compute 3rd-order neighbors (i→j→k→p)
  ↓
[2] Embedding Layer: x (5D) → x (128D)
  ↓
[3] SGMP Interactions (repeat 3 times):
    - Compute geometric features (distances, angles, dihedral)
    - Message passing with geometric encoding
    - Residual connection (x = x + update)
  ↓
[4] Global Pooling: Node features → Graph-level features
  ↓
[5] Classification Head:
    - Linear(128 → 64)
    - ReLU
    - Linear(64 → 2)
  ↓
Output: Logits [batch_size, num_classes]
```

---

## Detailed Breakdown

### **Step 1: Graph Preparation** (in training script)

```python
# Start with edge_index (2-hop neighborhood)
edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

# Compute 3rd-order neighbors (i→j→k→p paths)
_, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
    edge_index, num_nodes, order=3
)
# edge_index_3rd shape: [4, num_paths]
# Contains indices: [i, j, k, p] for each path i→j→k→p
```

**What is edge_index_3rd?**
- It's a tensor containing all 3rd-order paths in the graph
- Each column represents one path: node i → j → k → p
- Example: `[[0, 1, 2, 3], [1, 2, 3, 4], ...]`
- This captures higher-order graph structure beyond immediate neighbors

---

### **Step 2: Embedding** (SGMP.py line 69)

```python
self.embedding = Sequential(
    Linear(input_channels_node, hidden_channels),  # 5 → 128
    ReLU(),
    Linear(hidden_channels, hidden_channels)       # 128 → 128
)

x = self.embedding(x)  # Transform initial node features
```

**Purpose:** Transform raw node features (RGB, area, shape) into a richer representation.

---

### **Step 3: Geometric Feature Extraction** (SGMP.py lines 71-91)

For each path i→j→k→p, compute:

#### **3.1 Distances** (3 distances per path)
```python
i, j, k, p = edge_index_3rd  # Unpack path indices

distances[1] = ||pos[j] - pos[i]||  # Distance from i to j
distances[2] = ||pos[k] - pos[j]||  # Distance from j to k
distances[3] = ||pos[p] - pos[j]||  # Distance from j to p
```

#### **3.2 Angles** (2 angles per path)
```python
# Angle between edges (i→j) and (j→k)
theta_ijk = arctan2(||(i→j) × (j→k)||, (i→j) · (j→k))

# Angle between edges (i→j) and (j→p)
theta_ijp = arctan2(||(i→j) × (j→p)||, (i→j) · (j→p))
```

#### **3.3 Dihedral Angle** (1 dihedral per path)
```python
# Angle between two planes defined by triangles ijk and ijp
v1 = (i→j) × (j→k)  # Normal to plane ijk
v2 = (i→j) × (j→p)  # Normal to plane ijp
phi_ijkp = arctan2(||v1 × v2||, v1 · v2) * sign
```

**Visual:**
```
    k
   /|
  / |
 j  | <- theta_ijk (angle at j)
 |\ |
 | \|
 i  p
    ↑
    phi_ijkp (dihedral angle between planes ijk and ijp)
```

**Why these features?**
- **Distances:** Capture scale and local density
- **Angles:** Capture local geometry and curvature
- **Dihedral angles:** Capture 3D structure and chirality

---

### **Step 4: SGMP Interactions** (3 layers, SGMP.py lines 93-100)

Each interaction layer (SPNN) does:

#### **4.1 Encode Geometric Features** (SPNN.py lines 181-189)

```python
# 1st-order: Just distance
geo_encoding_1st = MLP(GaussianSmearing(distance_ij))

# 2nd-order: Distance + angle
geo_encoding_2nd = MLP(concat[
    GaussianSmearing(distance_jk),
    GaussianSmearing(theta_ijk)
])

# 3rd-order: Distance + angle + dihedral
geo_encoding_3rd = MLP(concat[
    GaussianSmearing(distance_jp),
    GaussianSmearing(theta_ijp),
    GaussianSmearing(phi_ijkp)
])
```

**What is GaussianSmearing?**
- Converts scalar values into high-dimensional vectors
- Like a "soft one-hot encoding" for continuous values
- Example: distance 2.5 → 50D vector with Gaussian bumps around 2.5
- Helps neural network learn smooth distance/angle functions

#### **4.2 Combine Node Features + Geometry** (SPNN.py lines 191-203)

```python
# For each path i→j→k→p, concatenate:
message = concat[
    x[i],               # Features of node i (128D)
    x[j],               # Features of node j (128D)
    x[k],               # Features of node k (128D)
    x[p],               # Features of node p (128D)
    geo_encoding_1st,   # Distance i→j (128D)
    geo_encoding_2nd,   # Distance+angle j→k (128D)
    geo_encoding_3rd,   # Distance+angle+dihedral j→p (128D)
]  # Total: 7 × 128D = 896D

# Process with MLP
update = MLP(message)  # 896D → 128D
```

**Why 7 components?**
- 4 node features (i, j, k, p)
- 3 geometric encodings (1st, 2nd, 3rd order)

#### **4.3 Aggregate to Nodes** (SPNN.py line 206)

```python
# Aggregate all messages arriving at node i
x_new[i] = sum of all updates from paths starting at i

# In code:
x = scatter(update, i, dim=0, reduce='add')
```

#### **4.4 Residual Connection** (SGMP.py line 94)

```python
x = x + interaction(x, ...)  # Add update to original features
```

**Why residual?**
- Helps gradients flow during training
- Preserves original information while adding new insights

---

### **Step 5: Global Pooling** (SGMP.py line 105)

```python
# Aggregate all node features to graph-level
x = scatter(x, batch, dim=0, reduce='add')  # Sum pooling

# Example:
# Batch: [0, 0, 0, 1, 1, 1, 1, 2, 2]  (3 graphs)
# x: [node0_feat, node1_feat, ..., node8_feat]
# Output: [graph0_feat, graph1_feat, graph2_feat]
```

**Alternatives:**
- `reduce='add'`: Sum all node features (default)
- `reduce='mean'`: Average node features
- `reduce='max'`: Max-pool node features

**This is your "latent" representation!** Shape: [batch_size, hidden_channels]

---

### **Step 6: Classification Head** (SGMP.py lines 106-108)

```python
x = Linear(128 → 64)(x)
x = ReLU(x)
x = Linear(64 → 2)(x)  # Final output
```

Simple 2-layer MLP for classification.

---

## Key Questions for Your Professor

### **1. About the "Decoder" Mention**

**Question:** _"You mentioned SGMP was used with a decoder. Looking at the code (`main_base_2.py` and `SGMP.py`), I only see the classification head (lin1 + lin2 layers). Could you clarify:_
- _Was there a separate decoder model used in the original work?_
- _Or are you referring to using the pooled features (line 105) as the 'latent' for downstream analysis?_
- _If there was a decoder, was it for reconstruction, generation, or perturbation visualization?"_

**Why ask:** The code doesn't show a decoder, but professors sometimes use the term loosely or there might be unpublished work.

---

### **2. About Latent Space for Explainability**

**Question:** _"For my explainability goals (perturbation analysis), should I extract the latent representation from:_
- _A) The pooled features after line 105 (before classification layers)?_
- _B) The intermediate features after lin1 (the 64D vector)?_
- _C) Or implement a separate autoencoder-style architecture?"_

**Why ask:** Clarifies where to extract features for your perturbation experiments.

---

### **3. About Training Approach**

**Question:** _"I see two possible approaches:_
- _A) Train SGMP end-to-end for classification (as in `main_base_2.py`)_
- _B) Add a separate encoder-decoder architecture with reconstruction loss_

_Which approach was used in the original geometric tree paper? And which would you recommend for interpretability?"_

**Why ask:** Determines if you should follow the existing code exactly or build something new.

---

### **4. About the 3rd-Order Neighbors**

**Question:** _"The SGMP model uses 3rd-order paths (i→j→k→p). This requires:_
- _Computing all possible 3-hop paths in the graph_
- _Calculating geometric features (distances, angles, dihedrals) for each path_

_For DRAC images with ~100 nodes, this creates many paths. Are there computational tricks or approximations used in practice?"_

**Why ask:** Understanding scalability and if there are optimizations you should know about.

---

### **5. About Healthy vs Unhealthy Training**

**Question:** _"For my dataset (balanced 50/50 healthy vs unhealthy), should I:_
- _A) Train on both classes jointly (standard supervised learning)_
- _B) Train only on healthy samples (anomaly detection approach)_
- _C) Pre-train on both, then fine-tune on specific subsets?_

_What approach gives the most interpretable latent space for perturbation analysis?"_

**Why ask:** Gets their recommendation based on experience.

---

### **6. About Perturbation Visualization**

**Question:** _"Once I have the latent representation and identify discriminative dimensions, how should I visualize perturbations?_
- _A) Decode latent → graph → image (requires decoder)_
- _B) Analyze changes in graph properties (node features, edge patterns)_
- _C) Use attention/saliency maps on the original images_
- _D) Something else?_

_What did the original geometric tree work use for visualization?"_

**Why ask:** Practical guidance on the visualization component.

---

## Understanding the Code Flow

### **Complete Training Loop** (from `main_base_2.py`)

```python
# 1. Load batch
x, pos, edge_index, batch, y = data

# 2. Prepare graph
edge_index, _ = add_self_loops(edge_index, num_nodes)
_, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
    edge_index, num_nodes, order=3
)

# 3. Forward pass
out = model(x, pos, batch, edge_index_3rd)  # [batch, 2]

# 4. Loss
loss = CrossEntropyLoss(out, y)

# 5. Backward
loss.backward()
optimizer.step()
```

**Simple and direct!** No separate encoder/decoder.

---

## What's NOT in the Current Code

Things that might be useful but aren't implemented:

1. **Explicit latent extraction** - Would need to modify forward() to return intermediate features
2. **Graph decoder** - No way to reconstruct graphs from latent vectors
3. **Perturbation tools** - No built-in way to perturb and visualize
4. **Attention mechanisms** - No interpretability built in
5. **Contrastive learning** - No self-supervised pretraining

---

## Summary for Professor Discussion

**What the current code does:**
- ✅ SGMP as end-to-end classifier
- ✅ 3rd-order geometric message passing
- ✅ Direct supervision for classification
- ✅ Standard training loop

**What's missing for your goals:**
- ❌ Explicit latent extraction
- ❌ Graph decoder for visualization
- ❌ Perturbation analysis tools
- ❌ Interpretability mechanisms

**Your question to professor:**
_"Should I extend the existing SGMP code with latent extraction and visualization tools, or is there a different architecture/approach you had in mind based on the geometric tree work?"_

---

## Code Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| SGMP model definition | `models/SGMP.py` | 24-116 |
| SPNN interaction layer | `models/SGMP.py` | 118-208 |
| Training loop | `main_base_2.py` | 759-787 |
| Evaluation | `main_base_2.py` | 789-842 |
| 3rd-order neighbor computation | `utils/utils.py` | (imported) |
| Geometric angle computation | `models/SGMP.py` | 8-10 |
| Gaussian smearing | `models/SGMP.py` | 12-21 |
| Pooling operation | `models/SGMP.py` | 105 |
| Classification head | `models/SGMP.py` | 106-108 |

---

This should give you a solid foundation for discussing with your professor! 🎯
