"""
Training script for DRAC using SGMP model directly
Aligned with original implementation in main_base_2.py
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
from models.SGMP import SGMP
from utils.utils import add_self_loops, find_higher_order_neighbors


def get_args():
    parser = argparse.ArgumentParser(description='Train DRAC with SGMP')

    # Data parameters
    parser.add_argument('--drac_root', type=str, required=True,
                       help='Root directory of DRAC dataset')
    parser.add_argument('--split', type=str, default='811',
                       help='Train/valid/test split (e.g., 811 for 80/10/10)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')

    # Model parameters (matching original SGMP usage)
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='Hidden dimension size')
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
    parser.add_argument('--test_every', type=int, default=5,
                       help='Test every N epochs')

    # Experiment parameters
    parser.add_argument('--save_dir', type=str, default='./results/drac_sgmp',
                       help='Directory to save results')
    parser.add_argument('--healthy_only', action='store_true',
                       help='Train only on healthy samples')

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

        # Add self-loops and compute 3rd order neighbors (matching original implementation)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
            edge_index, num_nodes, order=3
        )

        # Forward pass
        optimizer.zero_grad()
        out = model(x, pos, batch, edge_index_3rd)

        # Compute loss
        loss = criterion(out, y.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        # Track predictions
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model (matching original implementation)"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

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
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes, fill_value=-1.)
            _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
                edge_index, num_nodes, order=3
            )

            # Forward pass
            out = model(x, pos, batch, edge_index_3rd)

            # Compute loss
            loss = criterion(out, y.reshape(-1))
            total_loss += loss.item() * data.num_graphs

            # Track predictions
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

            # Compute probabilities (matching original normalization method)
            out_data = out.detach().cpu().numpy()
            min_values = np.min(out_data, axis=1)
            temp = out_data - min_values[:, np.newaxis]
            row_sums = temp.sum(axis=1)
            normalized_out = temp / row_sums[:, np.newaxis]

            for i in normalized_out:
                all_probs.append(list(i))

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute ROC-AUC
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    return avg_loss, accuracy, roc_auc, all_preds, all_labels, all_probs


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

    # Create model (matching original implementation exactly)
    print("\n" + "="*80)
    print("Initializing SGMP Model")
    print("="*80)

    model = SGMP(
        input_channels_node=input_channels,
        hidden_channels=args.hidden_channels,
        output_channels=2,  # Binary classification
        num_interactions=args.num_interactions,
        cutoff=args.cutoff,
        readout=args.readout
    ).to(device)

    print(f"Model: SGMP")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Num interactions: {args.num_interactions}")
    print(f"  Readout: {args.readout}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss (matching original implementation)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=10, min_lr=5e-5
    )
    criterion = nn.CrossEntropyLoss()

    # Logging
    log_file = os.path.join(args.save_dir, 'log.csv')
    result_file = os.path.join(args.save_dir, 'results.txt')

    with open(log_file, 'w') as f:
        f.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Val ROC-AUC,Time\n")

    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80)

    best_val_acc = 0
    best_val_roc = 0
    best_epoch = 0
    best_model_state = None

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate periodically
        if epoch % args.test_every == 0:
            val_loss, val_acc, val_roc, _, _, _ = evaluate(model, valid_loader, criterion, device)

            # Step scheduler
            if epoch >= 100:
                scheduler.step(val_loss)

            elapsed = time.time() - start_time

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ROC-AUC: {val_roc:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # Log to file
            with open(log_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_roc:.4f},{elapsed:.1f}\n")

            # Save best model (matching original: best validation accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_roc = val_roc
                best_epoch = epoch
                best_model_state = model.state_dict().copy()

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
    model.load_state_dict(best_model_state)

    # Evaluate on all splits
    train_loss, train_acc, train_roc, train_preds, train_labels, _ = evaluate(
        model, train_loader, criterion, device
    )
    val_loss, val_acc, val_roc, val_preds, val_labels, _ = evaluate(
        model, valid_loader, criterion, device
    )
    test_loss, test_acc, test_roc, test_preds, test_labels, _ = evaluate(
        model, test_loader, criterion, device
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

    # Save results (matching original format)
    with open(result_file, 'w') as f:
        f.write(f"Final, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Train roc: {train_roc:.4f}, "
                f"Valid loss: {val_loss:.4f}, Valid acc: {val_acc:.4f}, Valid roc: {val_roc:.4f}, "
                f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}, Test roc: {test_roc:.4f}\n\n")

        f.write(f"Best Model (Epoch {best_epoch}), Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Train roc: {train_roc:.4f}, "
                f"Valid loss: {val_loss:.4f}, Valid acc: {val_acc:.4f}, Valid roc: {val_roc:.4f}, "
                f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}, Test roc: {test_roc:.4f}\n\n")

        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n")
        f.write(classification_report(test_labels, test_preds, target_names=['Healthy', 'Unhealthy']))

    print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    main()
