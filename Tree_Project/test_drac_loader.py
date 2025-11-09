"""
Test script for DRAC data loader
Tests image-to-graph conversion and data loading
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append('.')

from drac_data_loader_filtered import DRACDataset, load_drac_data


def test_single_image(dataset, idx=0):
    """
    Test single image conversion to graph
    """
    print(f"\n{'='*50}")
    print(f"Testing Single Image Conversion")
    print(f"{'='*50}")
    
    # Get a single graph
    data = dataset.get(idx)
    
    print(f"\nGraph Statistics:")
    print(f"  Number of nodes: {data.x.shape[0]}")
    print(f"  Node feature dimension: {data.x.shape[1]}")
    print(f"  Number of edges: {data.edge_index.shape[1]}")
    print(f"  Position shape: {data.pos.shape}")
    print(f"  Label: {data.y.item()}")
    
    print(f"\nNode Features:")
    print(f"  Mean: {data.x.mean(dim=0).numpy()}")
    print(f"  Std: {data.x.std(dim=0).numpy()}")
    print(f"  Min: {data.x.min(dim=0).values.numpy()}")
    print(f"  Max: {data.x.max(dim=0).values.numpy()}")
    
    print(f"\nPosition Statistics:")
    print(f"  X range: [{data.pos[:, 0].min():.3f}, {data.pos[:, 0].max():.3f}]")
    print(f"  Y range: [{data.pos[:, 1].min():.3f}, {data.pos[:, 1].max():.3f}]")
    print(f"  Z range: [{data.pos[:, 2].min():.3f}, {data.pos[:, 2].max():.3f}]")
    
    return data


def visualize_graph(data, image_path, save_path='graph_visualization.png'):
    """
    Visualize the graph structure overlaid on the original image
    """
    print(f"\n{'='*50}")
    print(f"Visualizing Graph Structure")
    print(f"{'='*50}")
    
    # Load original image
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((512, 512))
    except:
        print(f"Warning: Could not load image from {image_path}")
        img = None
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original image
    if img is not None:
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
    # Plot graph structure
    ax = axes[1]
    if img is not None:
        ax.imshow(img, alpha=0.3)
    
    # Get node positions (denormalize to image coordinates)
    pos = data.pos.numpy()
    x_coords = pos[:, 0] * 512
    y_coords = pos[:, 1] * 512
    
    # Plot edges
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        ax.plot([x_coords[src], x_coords[dst]], 
               [y_coords[src], y_coords[dst]], 
               'b-', alpha=0.1, linewidth=0.5)
    
    # Plot nodes
    ax.scatter(x_coords, y_coords, c='red', s=30, alpha=0.6, zorder=5)
    
    ax.set_title(f'Graph Structure ({len(x_coords)} nodes, {edge_index.shape[1]} edges)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def test_data_loader(drac_root):
    """
    Test the full data loading pipeline
    """
    print(f"\n{'='*50}")
    print(f"Testing Data Loader Pipeline")
    print(f"{'='*50}")
    
    try:
        train_loader, valid_loader, test_loader = load_drac_data(
            drac_root, 
            split='811', 
            batch_size=4
        )
        
        print(f"\nData Loaders Created Successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Valid batches: {len(valid_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test loading a batch
        print(f"\nTesting Batch Loading...")
        for batch in train_loader:
            print(f"\nBatch Statistics:")
            print(f"  Number of graphs: {batch.num_graphs}")
            print(f"  Total nodes: {batch.x.shape[0]}")
            print(f"  Total edges: {batch.edge_index.shape[1]}")
            print(f"  Node features shape: {batch.x.shape}")
            print(f"  Positions shape: {batch.pos.shape}")
            print(f"  Labels: {batch.y.squeeze().numpy()}")
            
            # Check for NaN or Inf
            has_nan = torch.isnan(batch.x).any()
            has_inf = torch.isinf(batch.x).any()
            print(f"\n  Data Quality:")
            print(f"    Contains NaN: {has_nan}")
            print(f"    Contains Inf: {has_inf}")
            
            break
        
        return train_loader, valid_loader, test_loader
        
    except Exception as e:
        print(f"\nError during data loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_batch_consistency(loader, n_batches=5):
    """
    Test consistency across multiple batches
    """
    print(f"\n{'='*50}")
    print(f"Testing Batch Consistency")
    print(f"{'='*50}")
    
    node_counts = []
    edge_counts = []
    labels = []
    
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        
        node_counts.append(batch.x.shape[0])
        edge_counts.append(batch.edge_index.shape[1])
        labels.extend(batch.y.squeeze().cpu().numpy().tolist())
    
    print(f"\nStatistics across {n_batches} batches:")
    print(f"  Nodes per batch: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}")
    print(f"  Edges per batch: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}")
    print(f"  Label distribution: {np.bincount([int(l) for l in labels])}")


def main():
    """
    Main testing function
    """
    print("="*70)
    print("DRAC Data Loader Testing Suite")
    print("="*70)
    
    # Set paths
    drac_root = '../DRAC'
    
    import os
    train_img_dir = os.path.join(drac_root, 
                                 'C. Diabetic Retinopathy Grading',
                                 '1. Original Images',
                                 'a. Training Set')
    
    label_csv = os.path.join(drac_root,
                            'C. Diabetic Retinopathy Grading',
                            '2. Groundtruths',
                            'a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv')
    
    # Test 1: Create dataset
    print(f"\nTest 1: Creating Dataset")
    print(f"  Image directory: {train_img_dir}")
    print(f"  Label CSV: {label_csv}")
    
    try:
        dataset = DRACDataset(train_img_dir, label_csv, use_superpixels=True)
        print(f"  ✓ Dataset created successfully")
        print(f"  Total images: {len(dataset)}")
    except Exception as e:
        print(f"  ✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Single image conversion
    if len(dataset) > 0:
        print(f"\nTest 2: Single Image Conversion")
        try:
            data = test_single_image(dataset, idx=0)
            print(f"  ✓ Single image conversion successful")
            
            # Visualize if possible
            if len(dataset.image_files) > 0:
                visualize_graph(data, dataset.image_files[0])
                print(f"  ✓ Graph visualization created")
        except Exception as e:
            print(f"  ✗ Error in single image test: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Data loader pipeline
    print(f"\nTest 3: Data Loader Pipeline")
    train_loader, valid_loader, test_loader = test_data_loader(drac_root)
    
    if train_loader is not None:
        print(f"  ✓ Data loaders created successfully")
        
        # Test 4: Batch consistency
        print(f"\nTest 4: Batch Consistency")
        test_batch_consistency(train_loader, n_batches=5)
        print(f"  ✓ Batch consistency test passed")
    else:
        print(f"  ✗ Data loader creation failed")
    
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()