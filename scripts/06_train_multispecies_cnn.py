#!/usr/bin/env python3
"""
Train CNN classifier on multi-species V-gene dataset.

This script trains the same CNN architecture as script 04, but on a larger
multi-species dataset for improved generalization.

Usage:
    python scripts/06_train_multispecies_cnn.py \
        --train-csv data/processed/train_multispecies.csv \
        --val-csv data/processed/val_multispecies.csv \
        --output-dir models \
        --epochs 50
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.encoding import sequences_to_tensor
from src.models.classifier import VGeneCNN


class VGeneDataset(Dataset):
    """Dataset for V-gene sequences."""

    def __init__(self, csv_file, max_length=116):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        label = row['label']

        # Encode sequence
        tensor = sequences_to_tensor([sequence], max_length=self.max_length)[0]

        return tensor, torch.tensor(label, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Binary predictions
    binary_preds = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, auc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    binary_preds = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, auc


def plot_training_history(history, output_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training History - Multi-Species CNN', fontsize=16)

    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        ax.plot(history['epoch'], history[f'train_{metric}'], label='Train', marker='o')
        ax.plot(history['epoch'], history[f'val_{metric}'], label='Val', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN on multi-species V-gene dataset"
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/processed/train_multispecies.csv"),
        help="Training CSV file"
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/processed/val_multispecies.csv"),
        help="Validation CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for model"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=116,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    print("=" * 70)
    print("TRAIN MULTI-SPECIES V-GENE CNN")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading data...")
    print(f"   Train: {args.train_csv}")
    print(f"   Val: {args.val_csv}")

    train_dataset = VGeneDataset(args.train_csv, max_length=args.max_length)
    val_dataset = VGeneDataset(args.val_csv, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\n✅ Dataset loaded:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Model
    print(f"\n🏗️  Building model...")
    model = VGeneCNN(
        input_channels=20,
        seq_length=args.max_length,
        num_filters=[64, 128, 256],
        kernel_size=3,
        dropout=0.3
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    print(f"\n🏋️  Training for {args.epochs} epochs...")
    print(f"   Learning rate: {args.lr}")

    history = {
        'epoch': [],
        'train_loss': [], 'train_accuracy': [], 'train_precision': [],
        'train_recall': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_auc': []
    }

    best_val_f1 = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )

        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['train_precision'].append(train_prec)
        history['train_recall'].append(train_rec)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # Print progress
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch

            args.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.output_dir / "best_model_multispecies.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  💾 Best model saved (F1: {val_f1:.4f})")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n🏆 Best model:")
    print(f"   Epoch: {best_epoch}")
    print(f"   Val F1: {best_val_f1:.4f}")
    print(f"   Saved: {model_path}")

    # Save history
    history_df = pd.DataFrame(history)
    history_csv = Path("results") / "training_history_multispecies.csv"
    history_csv.parent.mkdir(exist_ok=True)
    history_df.to_csv(history_csv, index=False)
    print(f"\n📊 Training history saved: {history_csv}")

    # Plot
    plot_path = Path("results") / "training_history_multispecies.png"
    plot_training_history(history, plot_path)

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review training curves in results/training_history_multispecies.png")
    print("  2. Apply to new species:")
    print("     python scripts/12_run_tblastn.py  # (using best_model_multispecies.pt)")


if __name__ == "__main__":
    main()
