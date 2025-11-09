"""
Training script for DRAC binary classifier (healthy vs unhealthy)
Using SGMP encoder for geometric graph representation learning
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from drac_data_loader import load_drac_data
from models.graph_autoencoder import GraphClassifier
from utils.utils import add_self_loops, find_higher_order_neighbors


def get_args():
    parser = argparse.ArgumentParser(description='Train DRAC Classifier')

    # Data parameters
    parser.add_argument('--drac_root', type=str, required=True,
                       help='Root directory of DRAC dataset')
    parser.add_argument('--split', type=str, default='811',
                       help='Train/valid/test split (e.g., 811 for 80/10/10)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')

    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent space dimension')
    parser.add_argument('--num_interactions', type=int, default=3,
                       help='Number of SGMP interaction layers')
    parser.add_argument('--cutoff', type=float, default=10.0,
                       help='Cutoff distance for geometric features')
    parser.add_argument('--readout', type=str, default='add',
                       choices=['add', 'mean', 'sum'],
                       help='Graph readout function')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')

    # Experiment parameters
    parser.add_argument('--save_dir', type=str, default='./results/drac_classifier',
                       help='Directory to save results')
    parser.add_argument('--pretrained_autoencoder', type=str, default=None,
                       help='Path to pretrained autoencoder checkpoint (optional)')
    parser.add_argument('--test_every', type=int, default=5,
                       help='Test every N epochs')

    # Data filtering
    parser.add_argument('--healthy_only', action='store_true',
                       help='Train only on healthy samples (for anomaly detection experiments)')

    args = parser.parse_args()
    return args


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data in loader:
        x, pos, edge_index, batch, y = (
            data.x.float().to(device),
            data.pos.to(device),
            data.edge_index.to(device),
            data.batch.to(device),
            data.y.long().to(device)
        )

        num_nodes = data.num_nodes

        # Add self-loops and compute 3rd order neighbors for SGMP
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
            edge_index, num_nodes, order=3
        )

        # Forward pass
        optimizer.zero_grad()
        logits = model(x, pos, batch, edge_index_3rd)

        # Compute loss
        loss = criterion(logits, y.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        # Track predictions
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, return_predictions=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_latents = []

    with torch.no_grad():
        for data in loader:
            x, pos, edge_index, batch, y = (
                data.x.float().to(device),
                data.pos.to(device),
                data.edge_index.to(device),
                data.batch.to(device),
                data.y.long().to(device)
            )

            num_nodes = data.num_nodes

            # Add self-loops and compute 3rd order neighbors
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
                edge_index, num_nodes, order=3
            )

            # Forward pass with latent extraction
            logits, z = model(x, pos, batch, edge_index_3rd, return_latent=True)

            # Compute loss
            loss = criterion(logits, y.reshape(-1))
            total_loss += loss.item() * data.num_graphs

            # Track predictions
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs)
            all_latents.append(z.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute ROC-AUC (binary classification)
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    if return_predictions:
        all_latents = np.vstack(all_latents)
        return avg_loss, accuracy, roc_auc, all_preds, all_labels, all_probs, all_latents

    return avg_loss, accuracy, roc_auc


def main():
    args = get_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("\n" + "="*80)
    print("Loading DRAC Dataset")
    print("="*80)

    filter_labels = [0] if args.healthy_only else None

    train_loader, valid_loader, test_loader = load_drac_data(
        drac_root=args.drac_root,
        split=args.split,
        batch_size=args.batch_size,
        filter_labels=filter_labels,
        binary_classification=True
    )

    print(f"\nDataset splits: Train={len(train_loader.dataset)}, "
          f"Valid={len(valid_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Get input dimensions from first batch
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch.x.shape[1]
    print(f"Input node features: {input_channels}")

    # Create model
    print("\n" + "="*80)
    print("Initializing Model")
    print("="*80)

    model = GraphClassifier(
        input_channels_node=input_channels,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        num_classes=2,  # Binary classification
        num_interactions=args.num_interactions,
        cutoff=args.cutoff,
        readout=args.readout
    ).to(device)

    # Load pretrained encoder if specified
    if args.pretrained_autoencoder is not None:
        print(f"Loading pretrained encoder from {args.pretrained_autoencoder}")
        checkpoint = torch.load(args.pretrained_autoencoder, map_location=device)
        model.load_pretrained_encoder(checkpoint['model_state_dict'])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80)

    best_val_acc = 0
    best_val_roc = 0
    best_epoch = 0
    results = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate periodically
        if epoch % args.test_every == 0:
            val_loss, val_acc, val_roc = evaluate(model, valid_loader, criterion, device)
            scheduler.step(val_loss)

            elapsed = time.time() - start_time

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ROC-AUC: {val_roc:.4f} | "
                  f"Time: {elapsed:.1f}s")

            results.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_roc': val_roc,
                'time': elapsed
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_roc = val_roc
                best_epoch = epoch

                checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_roc': val_roc,
                    'args': args
                }, checkpoint_path)

    # Final evaluation on best model
    print("\n" + "="*80)
    print("Final Evaluation (Best Model)")
    print("="*80)

    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on all splits
    train_loss, train_acc, train_roc, train_preds, train_labels, train_probs, train_latents = evaluate(
        model, train_loader, criterion, device, return_predictions=True
    )
    val_loss, val_acc, val_roc, val_preds, val_labels, val_probs, val_latents = evaluate(
        model, valid_loader, criterion, device, return_predictions=True
    )
    test_loss, test_acc, test_roc, test_preds, test_labels, test_probs, test_latents = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )

    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ROC-AUC: {train_roc:.4f}")
    print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC-AUC: {val_roc:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, ROC-AUC: {test_roc:.4f}")

    # Confusion matrix
    print("\nTest Set Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    print("\nTest Set Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Healthy', 'Unhealthy']))

    # Save results
    results_file = os.path.join(args.save_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"DRAC Classifier Results\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Best Model (Epoch {best_epoch}):\n")
        f.write(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ROC-AUC: {train_roc:.4f}\n")
        f.write(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC-AUC: {val_roc:.4f}\n")
        f.write(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, ROC-AUC: {test_roc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n")
        f.write(classification_report(test_labels, test_preds, target_names=['Healthy', 'Unhealthy']))

    # Save latent representations for analysis
    latents_file = os.path.join(args.save_dir, 'latent_representations.npz')
    np.savez(latents_file,
             train_latents=train_latents,
             train_labels=train_labels,
             val_latents=val_latents,
             val_labels=val_labels,
             test_latents=test_latents,
             test_labels=test_labels)

    print(f"\nResults saved to {args.save_dir}")
    print(f"Latent representations saved to {latents_file}")


if __name__ == '__main__':
    main()
