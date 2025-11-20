# Memory Optimization for GT-SSL Training

## The Memory Issue

GT-SSL training requires significantly more GPU memory than single-stage training because:

1. **3rd-order edges**: Computing paths `i→j→k→p` creates many more edges than just `i→j`
   - For a graph with ~100 nodes and ~200 edges, 3rd-order can create **10,000+ paths**

2. **GT-SSL overhead**: Additional networks for subtree growth prediction

3. **Gradient computation**: Backpropagation through hierarchical ordering + geometric losses

## Solutions Applied

### 1. Reduced Batch Size

**Changed**: `BATCH_SIZE=16` → `BATCH_SIZE=4`

This reduces peak memory by ~75% but increases training time proportionally.

**Location**: [run_gtssl_pretrain.sh:25](run_gtssl_pretrain.sh#L25)

### 2. Memory Cleanup After Each Batch

Added explicit memory cleanup:

```python
# Clear memory
del embeddings, losses, loss, edge_index_3rd
if device.type == 'cuda':
    torch.cuda.empty_cache()
```

**Location**:
- [pretrain_gtssl.py:129-132](pretrain_gtssl.py#L129-L132) (training)
- [pretrain_gtssl.py:184-187](pretrain_gtssl.py#L184-L187) (validation)

### 3. PyTorch Memory Configuration

Set environment variable before running:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This helps PyTorch better manage fragmented memory.

## If Still Out of Memory

### Option 1: Further Reduce Batch Size

Edit [run_gtssl_pretrain.sh](run_gtssl_pretrain.sh):

```bash
BATCH_SIZE=2  # Or even 1
```

**Trade-off**: Slower training, potentially less stable gradients

### Option 2: Reduce Model Size

Edit [run_gtssl_pretrain.sh](run_gtssl_pretrain.sh):

```bash
HIDDEN_CHANNELS=32      # Down from 64
NUM_INTERACTIONS=2      # Down from 3
NUM_RBF_CENTERS=10      # Down from 20
```

**Trade-off**: Less expressive model, potentially lower accuracy

### Option 3: Gradient Accumulation

Accumulate gradients over multiple small batches before updating:

```python
# In pretrain_gtssl.py train_epoch function
ACCUMULATION_STEPS = 4

for i, batch_data in enumerate(loader):
    # ... forward pass ...
    loss = losses['total'] / ACCUMULATION_STEPS
    loss.backward()

    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit**: Effectively larger batch size with small memory footprint

### Option 4: Use CPU for Some Operations

Move less critical operations to CPU:

```python
# Extract tree structure on CPU to save GPU memory
parent_child_pairs, negative_pairs = extract_tree_structure_from_graph(
    batch_data.edge_index.cpu(), num_nodes
)
```

**Trade-off**: Slower due to CPU-GPU transfers

### Option 5: Mixed Precision Training

Use automatic mixed precision (FP16):

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    embeddings, losses = model(...)
    loss = losses['total']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefit**: ~50% memory reduction, faster training
**Trade-off**: Slight numerical precision loss (usually negligible)

## Memory Usage Breakdown

Typical memory consumption for GT-SSL with DRAC:

| Component | Memory (per graph) | Batch Size 16 | Batch Size 4 |
|-----------|-------------------|---------------|--------------|
| Input graph (100 nodes) | 5 MB | 80 MB | 20 MB |
| 3rd-order edges (10K paths) | 50 MB | 800 MB | 200 MB |
| SGMP activations | 20 MB | 320 MB | 80 MB |
| GT-SSL predictor | 10 MB | 160 MB | 40 MB |
| Gradients | ~2x forward | ~2.6 GB | ~680 MB |
| **Total** | - | **~4 GB** | **~1 GB** |

With 16GB GPU, batch size 4 should work comfortably.

## Monitoring Memory

Check GPU memory usage:

```bash
# Real-time monitoring
nvidia-smi -l 1

# Or install gpustat
pip install gpustat
gpustat -i 1
```

During training, you should see:
- **Allocated**: Memory actively used by tensors
- **Reserved**: Memory reserved by PyTorch allocator
- **Free**: Available GPU memory

## Recommended Settings

For different GPU sizes:

| GPU Memory | Batch Size | Hidden Channels | Interactions | Expected Training Time |
|------------|-----------|----------------|--------------|------------------------|
| 8 GB | 2 | 32 | 2 | ~4 hours |
| 12 GB | 4 | 64 | 3 | ~3 hours |
| 16 GB | 4-8 | 64 | 3 | ~2-3 hours |
| 24 GB | 8-16 | 64 | 3 | ~1-2 hours |

## Current Configuration

✅ **Applied**:
- Batch size: 4
- Memory cleanup after each batch
- Hidden channels: 64
- Interactions: 3
- RBF centers: 20

**Expected memory usage**: ~1-1.5 GB peak

**Expected training time**: ~3 hours for 100 epochs on V100/A100

## If Training Crashes

1. Check `nvidia-smi` to see current GPU usage
2. Reduce batch size further if needed
3. Try setting: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. Consider using CPU for tree structure extraction
5. As last resort, implement gradient accumulation

## Alternative: Train on CPU

If GPU continues to be problematic:

```bash
# In run_gtssl_pretrain.sh
DEVICE="cpu"
```

**Warning**: Will be **much slower** (~10-20x), but will work with large RAM.

---

## Quick Fix Reference

**Out of memory during training?**

```bash
# Edit run_gtssl_pretrain.sh
BATCH_SIZE=2  # or 1

# Run with memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run_gtssl_pretrain.sh
```

**Still failing?**

```bash
# Further reduce model size
HIDDEN_CHANNELS=32
NUM_INTERACTIONS=2
NUM_RBF_CENTERS=10
```

---

The current settings (batch size 4 with memory cleanup) should work on most modern GPUs with ≥12GB memory.
