#!/usr/bin/env python3
"""
Script 03: Train V-Gene Classifier
Version: 2.0.0
Purpose: Train CNN with terminal-region + dipeptide encoding for multiclass V-gene classification

ENCODING METHOD:
  - N-terminal 40 aa (one-hot encoded) = 800 features
  - C-terminal 40 aa (one-hot encoded) = 800 features
  - Dipeptide counts (full sequence) = 400 features
  Total: 2,000 fixed-length features (sequence-length invariant)

ARCHITECTURE:
  - 1D CNN treating the 2,000 features as a sequence
  - 3 convolutional blocks with batch normalization
  - Fully connected layers with dropout
  - ~8.4M parameters

CLASSES (5 total):
  0 = background, 1 = IGHV, 2 = IGKV, 3 = TRAV, 4 = TRBV

USAGE:
    python scripts/03_train_model.py \
        --train-csv data/processed_v2/train_multispecies_multiclass.csv \
        --val-csv data/processed_v2/val_multispecies_multiclass.csv \
        --output-dir models/v2_terminal_encoding \
        --epochs 30

Author: Luis Rana Cortizo
Date: 2026-02-21
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import terminal encoding and model from src
from src.features.terminal_encoding import TerminalRegionEncoder
from src.models.cnn_terminal import CNN_TerminalEncoding

# Class names
CLASS_NAMES = ['background', 'IGHV', 'IGKV', 'TRAV', 'TRBV']


class VGeneDataset(Dataset):
    """Dataset using terminal-region encoding."""

    def __init__(self, csv_file: Path):
        self.data = pd.read_csv(csv_file)
        self.encoder = TerminalRegionEncoder()

        self.sequences = self.data['sequence'].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Encode with terminal-region method (returns 2000 features)
        encoding = self.encoder.encode(seq)

        return torch.FloatTensor(encoding), torch.LongTensor([label])[0]


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, all_preds, all_labels


def plot_training_history(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = list(range(1, len(history['train_loss']) + 1))

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', marker='o')
    axes[1].plot(epochs, history['val_acc'], label='Val', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   💾 Training plot saved: {output_dir / 'training_history.png'}")


def plot_confusion_matrix(labels, predictions, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')

    # Labels
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_title("Confusion Matrix - Validation Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   💾 Confusion matrix saved: {output_dir / 'confusion_matrix.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Train V-gene classifier with terminal-region encoding (v2.0.0)'
    )
    parser.add_argument('--train-csv', type=Path, required=True,
                       help='Training CSV file')
    parser.add_argument('--val-csv', type=Path, required=True,
                       help='Validation CSV file')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('models/v2_terminal_encoding'),
                       help='Output directory for models and plots')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V-GENE CLASSIFIER TRAINING - v2.0.0")
    print("=" * 80)
    print(f"Encoding: Terminal-region (N/C-term + dipeptides)")
    print(f"Architecture: 1D CNN")
    print(f"Classes: {args.num_classes} ({', '.join(CLASS_NAMES)})")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print()

    # Load datasets
    print("📖 LOADING DATA")
    print("-" * 80)
    train_dataset = VGeneDataset(args.train_csv)
    val_dataset = VGeneDataset(args.val_csv)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    print(f"   Train: {len(train_dataset):,} sequences")
    print(f"   Val:   {len(val_dataset):,} sequences")
    print()

    # Create model
    print("🔨 INITIALIZING MODEL")
    print("-" * 80)
    model = CNN_TerminalEncoding(input_size=2000, num_classes=args.num_classes)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("🚀 TRAINING")
    print("=" * 80)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None
    best_preds = None
    best_labels = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_preds = val_preds
            best_labels = val_labels
            print(f"✅ New best model! Val Acc: {val_acc:.2f}%")

    # Load best model
    model.load_state_dict(best_model_state)

    # Save model
    model_path = args.output_dir / 'best_model.pt'
    torch.save(model.state_dict(), model_path)

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {model_path}")
    print()

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(args.output_dir / 'training_history.csv', index=False)

    # Generate plots
    print("📊 GENERATING PLOTS")
    print("-" * 80)
    plot_training_history(history, args.output_dir)
    plot_confusion_matrix(best_labels, best_preds, args.output_dir)

    # Classification report
    print()
    print("📋 CLASSIFICATION REPORT")
    print("-" * 80)
    print(classification_report(best_labels, best_preds,
                                target_names=CLASS_NAMES, digits=4))

    print()
    print("=" * 80)
    print("✅ ALL DONE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print("   - best_model.pt")
    print("   - training_history.csv")
    print("   - training_history.png")
    print("   - confusion_matrix.png")
    print()


if __name__ == '__main__':
    main()
