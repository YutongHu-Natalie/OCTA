"""
STAGE 2: Fine-tune pretrained SGMP encoder for Classification
Loads the pretrained encoder (either from autoencoder or GT-SSL) and adds classification head

Two-stage approach:
  Stage 1: pretrain_gtssl.py (GT-SSL) OR pretrain_sgmp_autoencoder.py (reconstruction)
  Stage 2: finetune_sgmp_classifier.py (classification) ← This file
"""

import argparse
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from drac_data_loader import load_drac_data
from models.SGMP import SGMP
from utils.utils import add_self_loops, find_higher_order_neighbors


def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune SGMP Classifier')

    # Data parameters
    parser.add_argument('--drac_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='811')
    parser.add_argument('--batch_size', type=int, default=16)

    # Pretrained model
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='Path to pretrained autoencoder checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights during fine-tuning')

    # Fine-tuning parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (lower than pretrain)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_every', type=int, default=5)

    # Experiment parameters
    parser.add_argument('--save_dir', type=str, default='./results/finetune_classifier')

    args = parser.parse_args()
    return args


def train_epoch(model, loader, optimizer, criterion, device):
    """Train classifier for one epoch"""
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

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
            edge_index, num_nodes, order=3
        )

        optimizer.zero_grad()
        out = model(x, pos, batch, edge_index_3rd)

        loss = criterion(out, y.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate classifier"""
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

            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes, fill_value=-1.)
            _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(
                edge_index, num_nodes, order=3
            )

            out = model(x, pos, batch, edge_index_3rd)

            loss = criterion(out, y.reshape(-1))
            total_loss += loss.item() * data.num_graphs

            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

            # Probabilities
            out_data = out.detach().cpu().numpy()
            min_values = np.min(out_data, axis=1)
            temp = out_data - min_values[:, np.newaxis]
            row_sums = temp.sum(axis=1)
            normalized_out = temp / row_sums[:, np.newaxis]

            for i in normalized_out:
                all_probs.append(list(i))

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    return avg_loss, accuracy, roc_auc, all_preds, all_labels


def main():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("\n" + "="*80)
    print("STAGE 2: FINE-TUNING for Classification")
    print("="*80)

    train_loader, valid_loader, test_loader = load_drac_data(
        drac_root=args.drac_root,
        split=args.split,
        batch_size=args.batch_size,
        binary_classification=True
    )

    print(f"\nDataset: Train={len(train_loader.dataset)}, "
          f"Valid={len(valid_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Load pretrained model
    print("\n" + "="*80)
    print("Loading Pretrained Model")
    print("="*80)

    checkpoint = torch.load(args.pretrained_model, map_location=device)

    # Check if this is a namespace or dict
    if hasattr(checkpoint['args'], '__dict__'):
        pretrain_args = checkpoint['args']
    else:
        # Convert dict to namespace
        from argparse import Namespace
        pretrain_args = Namespace(**checkpoint['args'])

    print(f"Loaded pretrained model from epoch {checkpoint['epoch']}")
    print(f"  Pretrain val loss: {checkpoint['val_loss']:.4f}")

    # Get input channels from data
    sample_data = train_loader.dataset[0]
    input_channels = sample_data.x.shape[1]

    # Create model with same architecture
    model = SGMP(
        input_channels_node=input_channels,
        hidden_channels=pretrain_args.hidden_channels,
        output_channels=2,  # Binary classification
        num_interactions=pretrain_args.num_interactions,
        cutoff=pretrain_args.cutoff,
        readout=pretrain_args.readout
    ).to(device)

    # Load encoder weights from pretrained model
    pretrained_state = checkpoint['model_state_dict']

    # Detect if this is GT-SSL or autoencoder based
    is_gtssl = any(k.startswith('encoder.') for k in pretrained_state.keys())

    if is_gtssl:
        print("Detected GT-SSL pretrained model")
        # Extract encoder weights (GT-SSL wraps SGMP as model.encoder)
        encoder_weights = {}
        for k, v in pretrained_state.items():
            if k.startswith('encoder.'):
                # Remove 'encoder.' prefix
                new_key = k.replace('encoder.', '')
                encoder_weights[new_key] = v
    else:
        print("Detected direct encoder pretrained model")
        # Weights are already at the top level
        encoder_weights = pretrained_state

    # Load weights (except final classification layers)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in encoder_weights.items()
                      if k in model_dict and 'lin1' not in k and 'lin2' not in k}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(f"\nLoaded {len(pretrained_dict)} pretrained layers")
    print(f"Randomly initialized classification head (lin1, lin2)")

    # Freeze encoder if requested
    if args.freeze_encoder:
        print("\nFreezing encoder weights...")
        for name, param in model.named_parameters():
            if 'lin1' not in name and 'lin2' not in name:
                param.requires_grad = False
        print("Only training classification head")
    else:
        print("\nFine-tuning entire model (encoder + classifier)")

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # Logging
    log_file = os.path.join(args.save_dir, 'finetune_log.csv')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Val ROC-AUC,Time\n")

    # Training loop
    print("\n" + "="*80)
    print("Fine-tuning for Classification")
    print("="*80)

    best_val_acc = 0
    best_model_state = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % args.test_every == 0:
            val_loss, val_acc, val_roc, _, _ = evaluate(model, valid_loader, criterion, device)

            if epoch >= 20:
                scheduler.step(val_loss)

            elapsed = time.time() - start_time

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ROC: {val_roc:.4f} | "
                  f"Time: {elapsed:.1f}s")

            with open(log_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},"
                       f"{val_acc:.4f},{val_roc:.4f},{elapsed:.1f}\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_roc': val_roc,
                    'pretrained_from': args.pretrained_model,
                    'args': args
                }, os.path.join(args.save_dir, 'best_finetuned_model.pt'))

    # Final evaluation
    print("\n" + "="*80)
    print("Fine-tuning Complete!")
    print("="*80)

    model.load_state_dict(best_model_state)

    train_loss, train_acc, train_roc, train_preds, train_labels = evaluate(
        model, train_loader, criterion, device
    )
    val_loss, val_acc, val_roc, val_preds, val_labels = evaluate(
        model, valid_loader, criterion, device
    )
    test_loss, test_acc, test_roc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ROC: {train_roc:.4f}")
    print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC: {val_roc:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, ROC: {test_roc:.4f}")

    print("\nTest Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Healthy', 'Unhealthy']))

    # Save results
    result_file = os.path.join(args.save_dir, 'results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Two-Stage Training Results\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Stage 1: Pretrained from {args.pretrained_model}\n")
        f.write(f"Stage 2: Fine-tuned for classification\n\n")
        f.write(f"Best Model (Epoch {best_epoch}):\n")
        f.write(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ROC: {train_roc:.4f}\n")
        f.write(f"Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC: {val_roc:.4f}\n")
        f.write(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, ROC: {test_roc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n")
        f.write(classification_report(test_labels, test_preds, target_names=['Healthy', 'Unhealthy']))

    print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    main()
