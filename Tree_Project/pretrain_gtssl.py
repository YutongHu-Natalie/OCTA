"""
Stage 1: GT-SSL Pretraining

This implements the CORRECT two-stage approach from the paper:
- Uses GT-SSL (Geometric Tree Self-Supervised Learning)
- Combines partial ordering constraint + subtree growth learning
- NO traditional autoencoder reconstruction!

Based on the geometric tree paper's actual approach.
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from drac_data_loader import DRACDataset
from models.SGMP import SGMP
from models.gtssl import GTSSL, extract_tree_structure_from_graph


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 1: GT-SSL Pretraining')

    # Data
    parser.add_argument('--drac_root', type=str, required=True,
                       help='Path to DRAC directory')
    parser.add_argument('--split', type=str, default='811',
                       help='Train/val/test split (e.g., "811" for 80/10/10)')

    # Model architecture
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='Hidden channels in SGMP')
    parser.add_argument('--num_interactions', type=int, default=3,
                       help='Number of SGMP interaction layers')
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Distance cutoff for SGMP')
    parser.add_argument('--readout', type=str, default='add',
                       choices=['add', 'sum', 'mean'],
                       help='Graph readout operation')
    parser.add_argument('--num_rbf_centers', type=int, default=20,
                       help='Number of RBF centers for GT-SSL')

    # GT-SSL hyperparameters
    parser.add_argument('--delta_margin', type=float, default=1.0,
                       help='Margin δ for negative pairs in ordering constraint')
    parser.add_argument('--lambda_order', type=float, default=1.0,
                       help='Weight for ordering loss')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_every', type=int, default=5)

    # Saving
    parser.add_argument('--save_dir', type=str, default='./results/gtssl_pretrain',
                       help='Directory to save results')

    # Data options
    parser.add_argument('--train_on_healthy_only', action='store_true',
                       help='Train only on healthy samples (class 0)')

    return parser.parse_args()


def train_epoch(model, loader, optimizer, device, args):
    model.train()

    total_loss = 0
    total_ordering_loss = 0
    total_generative_loss = 0
    num_batches = 0

    for batch_data in loader:
        batch_data = batch_data.to(device)

        # Extract tree structure for this batch
        # Note: edge_index might not exist in DRAC data, construct from edge_index_3rd
        if hasattr(batch_data, 'edge_index') and batch_data.edge_index is not None:
            edge_index = batch_data.edge_index
        else:
            # Construct simple edge_index from edge_index_3rd
            # Use i→j connections from edge_index_3rd[0] → edge_index_3rd[1]
            if batch_data.edge_index_3rd is not None:
                i = batch_data.edge_index_3rd[0]
                j = batch_data.edge_index_3rd[1]
                # Create unique edges
                edge_index = torch.stack([i, j], dim=0)
                edge_index = torch.unique(edge_index, dim=1)
            else:
                edge_index = None

        # Extract parent-child and negative pairs
        num_nodes = batch_data.x.shape[0]
        parent_child_pairs, negative_pairs = extract_tree_structure_from_graph(
            edge_index, num_nodes
        )

        # Forward pass
        embeddings, losses = model(
            x=batch_data.x,
            pos=batch_data.pos,
            batch=batch_data.batch,
            edge_index_3rd=batch_data.edge_index_3rd,
            parent_child_pairs=parent_child_pairs,
            negative_pairs=negative_pairs,
            edge_index=edge_index
        )

        loss = losses['total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_ordering_loss += losses['ordering'].item()
        total_generative_loss += losses['generative'].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ordering = total_ordering_loss / num_batches
    avg_generative = total_generative_loss / num_batches

    return avg_loss, avg_ordering, avg_generative


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()

    total_loss = 0
    total_ordering_loss = 0
    total_generative_loss = 0
    num_batches = 0

    for batch_data in loader:
        batch_data = batch_data.to(device)

        # Extract tree structure
        if hasattr(batch_data, 'edge_index') and batch_data.edge_index is not None:
            edge_index = batch_data.edge_index
        else:
            if batch_data.edge_index_3rd is not None:
                i = batch_data.edge_index_3rd[0]
                j = batch_data.edge_index_3rd[1]
                edge_index = torch.stack([i, j], dim=0)
                edge_index = torch.unique(edge_index, dim=1)
            else:
                edge_index = None

        num_nodes = batch_data.x.shape[0]
        parent_child_pairs, negative_pairs = extract_tree_structure_from_graph(
            edge_index, num_nodes
        )

        # Forward
        embeddings, losses = model(
            x=batch_data.x,
            pos=batch_data.pos,
            batch=batch_data.batch,
            edge_index_3rd=batch_data.edge_index_3rd,
            parent_child_pairs=parent_child_pairs,
            negative_pairs=negative_pairs,
            edge_index=edge_index
        )

        total_loss += losses['total'].item()
        total_ordering_loss += losses['ordering'].item()
        total_generative_loss += losses['generative'].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ordering = total_ordering_loss / num_batches
    avg_generative = total_generative_loss / num_batches

    return avg_loss, avg_ordering, avg_generative


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("\nLoading DRAC data...")

    # Paths
    train_img_dir = os.path.join(args.drac_root, 'C. Diabetic Retinopathy Grading', '1. Original Images', 'a. Training Set')
    train_label_csv = os.path.join(args.drac_root, 'C. Diabetic Retinopathy Grading', '2. Groundtruths', 'a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv')
    val_img_dir = os.path.join(args.drac_root, 'C. Diabetic Retinopathy Grading', '1. Original Images', 'b. Testing Set')

    # Decide which classes to use
    if args.train_on_healthy_only:
        filter_labels = [0]  # Only healthy
        print("Training on HEALTHY samples only (anomaly detection style)")
    else:
        filter_labels = None  # All classes
        print("Training on ALL samples (healthy + unhealthy)")

    # Load datasets
    train_dataset = DRACDataset(
        image_dir=train_img_dir,
        label_csv=train_label_csv,
        use_superpixels=True,
        binary_classification=True,
        filter_labels=filter_labels
    )

    val_dataset = DRACDataset(
        image_dir=val_img_dir,
        label_csv=train_label_csv,
        use_superpixels=True,
        binary_classification=True,
        filter_labels=filter_labels
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Get input dimension
    sample_data = train_dataset[0]
    input_channels = sample_data.x.shape[1]
    print(f"Node feature dimension: {input_channels}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    # Create encoder (SGMP)
    # We'll use SGMP but ignore its final classification layers
    encoder = SGMP(
        input_channels_node=input_channels,
        hidden_channels=args.hidden_channels,
        output_channels=2,  # Dummy, won't be used
        num_interactions=args.num_interactions,
        cutoff=args.cutoff,
        readout=args.readout
    )

    # Create GT-SSL model
    model = GTSSL(
        encoder=encoder,
        hidden_dim=args.hidden_channels,
        num_rbf_centers=args.num_rbf_centers,
        delta_margin=args.delta_margin,
        lambda_order=args.lambda_order
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Encoder: SGMP")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  RBF centers: {args.num_rbf_centers}")
    print(f"  Delta margin: {args.delta_margin}")
    print(f"  Lambda order: {args.lambda_order}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    # Training loop
    print(f"\nStarting GT-SSL pretraining for {args.epochs} epochs...")
    print("=" * 80)

    best_val_loss = float('inf')
    results_log = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_order, train_gen = train_epoch(
            model, train_loader, optimizer, device, args
        )

        # Validate
        val_loss, val_order, val_gen = validate(
            model, val_loader, device, args
        )

        epoch_time = time.time() - start_time

        # Log
        log_str = (f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} (Order: {train_order:.4f}, Gen: {train_gen:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (Order: {val_order:.4f}, Gen: {val_gen:.4f}) | "
                  f"Time: {epoch_time:.1f}s")

        print(log_str)
        results_log.append(log_str)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'best_gtssl_model.pt'))

        # Save checkpoint periodically
        if epoch % args.test_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': vars(args)
    }, os.path.join(args.save_dir, 'final_gtssl_model.pt'))

    # Save results
    with open(os.path.join(args.save_dir, 'training_log.txt'), 'w') as f:
        f.write('\n'.join(results_log))

    print("\n" + "=" * 80)
    print("GT-SSL Pretraining Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {args.save_dir}")
    print("\nNext step: Fine-tune for classification using:")
    print("  python finetune_sgmp_classifier.py --pretrained_model ./results/gtssl_pretrain/best_gtssl_model.pt")
    print("=" * 80)


if __name__ == '__main__':
    main()
