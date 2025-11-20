"""
STAGE 1: Pretrain SGMP with Graph Autoencoder (Reconstruction)
Following the two-stage approach: pretrain → fine-tune

This script implements the MISSING pretrain stage mentioned by the professor.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_scatter import scatter

from drac_data_loader import load_drac_data
from models.SGMP import SGMP
from utils.utils import add_self_loops, find_higher_order_neighbors


class SGMPAutoencoder(nn.Module):
    """
    Graph Autoencoder using SGMP as encoder
    Stage 1: Pretrain with reconstruction loss
    """
    def __init__(self, input_channels_node=5, hidden_channels=64,
                 latent_dim=32, num_interactions=3, cutoff=10.0, readout='add'):
        super(SGMPAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # Encoder: SGMP without classification head
        # We'll use SGMP but extract features before final layers
        self.encoder = SGMP(
            input_channels_node=input_channels_node,
            hidden_channels=hidden_channels,
            output_channels=latent_dim,  # Output to latent space
            num_interactions=num_interactions,
            cutoff=cutoff,
            readout=readout
        )

        # Decoder: Reconstruct node features from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, input_channels_node),  # Back to original features
        )

    def encode(self, x, pos, batch, edge_index_3rd):
        """Encode graph to latent representation"""
        return self.encoder(x, pos, batch, edge_index_3rd)

    def decode(self, z, num_nodes_per_graph):
        """
        Decode latent to node features

        Args:
            z: Latent vectors [batch_size, latent_dim]
            num_nodes_per_graph: Number of nodes in each graph
        """
        # Expand latent to each node
        batch_size = z.shape[0]

        # Create batch assignment for nodes
        node_batch = []
        for i in range(batch_size):
            node_batch.extend([i] * num_nodes_per_graph[i])
        node_batch = torch.tensor(node_batch, device=z.device)

        # Repeat latent for each node in the graph
        z_expanded = z[node_batch]  # [total_nodes, latent_dim]

        # Decode to node features
        x_recon = self.decoder(z_expanded)  # [total_nodes, input_features]

        return x_recon

    def forward(self, x, pos, batch, edge_index_3rd, num_nodes_per_graph):
        """Full forward: encode + decode"""
        z = self.encode(x, pos, batch, edge_index_3rd)
        x_recon = self.decode(z, num_nodes_per_graph)
        return x_recon, z


def reconstruction_loss(x_original, x_reconstructed, batch):
    """
    Compute reconstruction loss (MSE per graph, then average)
    """
    # MSE loss for node features
    mse = nn.functional.mse_loss(x_reconstructed, x_original, reduction='none')

    # Average over features, then sum over nodes per graph
    mse = mse.mean(dim=1)  # [num_nodes]

    # Sum per graph
    loss_per_graph = scatter(mse, batch, dim=0, reduce='mean')

    # Average over batch
    return loss_per_graph.mean()


def get_args():
    parser = argparse.ArgumentParser(description='Pretrain SGMP Autoencoder')

    # Data parameters
    parser.add_argument('--drac_root', type=str, required=True,
                       help='Root directory of DRAC dataset')
    parser.add_argument('--split', type=str, default='811',
                       help='Train/valid/test split')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')

    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent space dimension')
    parser.add_argument('--num_interactions', type=int, default=3,
                       help='Number of SGMP layers')
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Cutoff for geometric features')
    parser.add_argument('--readout', type=str, default='add',
                       help='Graph readout function')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Pretraining epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--test_every', type=int, default=5,
                       help='Test every N epochs')

    # Experiment parameters
    parser.add_argument('--save_dir', type=str, default='./results/pretrain_autoencoder',
                       help='Directory to save pretrained model')
    parser.add_argument('--train_on_healthy_only', action='store_true',
                       help='Pretrain only on healthy samples (unsupervised)')

    args = parser.parse_args()
    return args


def train_epoch(model, loader, optimizer, device):
    """Train autoencoder for one epoch"""
    model.train()
    total_loss = 0

    for data in loader:
        x, pos, edge_index, batch = (
            data.x.float().to(device),
            data.pos.to(device),
            data.edge_index.to(device),
            data.batch.to(device)
        )

        num_nodes = data.num_nodes

        # Compute 3rd-order neighbors
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
            edge_index, num_nodes, order=3
        )

        # Get number of nodes per graph
        num_nodes_per_graph = scatter(
            torch.ones(num_nodes, device=device),
            batch,
            dim=0,
            reduce='sum'
        ).long()

        # Forward pass
        optimizer.zero_grad()
        x_recon, z = model(x, pos, batch, edge_index_3rd, num_nodes_per_graph)

        # Reconstruction loss
        loss = reconstruction_loss(x, x_recon, batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Evaluate autoencoder"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            x, pos, edge_index, batch = (
                data.x.float().to(device),
                data.pos.to(device),
                data.edge_index.to(device),
                data.batch.to(device)
            )

            num_nodes = data.num_nodes

            # Compute 3rd-order neighbors
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
                edge_index, num_nodes, order=3
            )

            # Get number of nodes per graph
            num_nodes_per_graph = scatter(
                torch.ones(num_nodes, device=device),
                batch,
                dim=0,
                reduce='sum'
            ).long()

            # Forward pass
            x_recon, z = model(x, pos, batch, edge_index_3rd, num_nodes_per_graph)

            # Loss
            loss = reconstruction_loss(x, x_recon, batch)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def main():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("\n" + "="*80)
    print("STAGE 1: PRETRAINING with Reconstruction")
    print("="*80)

    filter_labels = [0] if args.train_on_healthy_only else None

    train_loader, valid_loader, test_loader = load_drac_data(
        drac_root=args.drac_root,
        split=args.split,
        batch_size=args.batch_size,
        filter_labels=filter_labels,
        binary_classification=True
    )

    print(f"\nDataset: Train={len(train_loader.dataset)}, "
          f"Valid={len(valid_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Get dimensions
    sample = next(iter(train_loader))
    input_channels = sample.x.shape[1]

    # Create model
    print("\n" + "="*80)
    print("Creating SGMP Autoencoder")
    print("="*80)

    model = SGMPAutoencoder(
        input_channels_node=input_channels,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        num_interactions=args.num_interactions,
        cutoff=args.cutoff,
        readout=args.readout
    ).to(device)

    print(f"Architecture:")
    print(f"  Input features: {input_channels}")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Num interactions: {args.num_interactions}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # Logging
    log_file = os.path.join(args.save_dir, 'pretrain_log.csv')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train Loss,Val Loss,Time\n")

    # Training loop
    print("\n" + "="*80)
    print("Pretraining with Reconstruction Loss")
    print("="*80)

    best_val_loss = float('inf')
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        if epoch % args.test_every == 0:
            val_loss = evaluate(model, valid_loader, device)
            scheduler.step(val_loss)

            elapsed = time.time() - start_time

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

            with open(log_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{elapsed:.1f}\n")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': args
                }, os.path.join(args.save_dir, 'pretrained_autoencoder.pt'))

    # Final evaluation
    print("\n" + "="*80)
    print("Pretraining Complete!")
    print("="*80)

    test_loss = evaluate(model, test_loader, device)

    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")

    print(f"\nPretrained model saved to:")
    print(f"  {os.path.join(args.save_dir, 'pretrained_autoencoder.pt')}")

    print(f"\nNext step: Fine-tune with classification using:")
    print(f"  python finetune_sgmp_classifier.py --pretrained_model {os.path.join(args.save_dir, 'pretrained_autoencoder.pt')}")


if __name__ == '__main__':
    main()
