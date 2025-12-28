#!/usr/bin/env python3
"""
Train multiclass CNN for V-gene classification.
Classes: 0=background, 1=IGHV, 2=IGKV, 3=TRAV, 4=TRBV
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.encoding import sequences_to_tensor
from src.models.classifier import VGeneCNN

# Class names
CLASS_NAMES = ['background', 'IGHV', 'IGKV', 'TRAV', 'TRBV']

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).long()  # CrossEntropyLoss expects long

        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }

def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).long()

            # Forward
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }

def plot_training_history(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val', marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['epoch'], history['train_acc'], label='Train', marker='o')
    axes[0, 1].plot(history['epoch'], history['val_acc'], label='Val', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score
    axes[1, 0].plot(history['epoch'], history['train_f1'], label='Train', marker='o')
    axes[1, 0].plot(history['epoch'], history['val_f1'], label='Val', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate (if available)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_multiclass.png'), dpi=300)
    plt.close()

    print(f"Saved training plot to {output_dir}/training_history_multiclass.png")

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
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_title("Confusion Matrix - Validation Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_multiclass.png'), dpi=300)
    plt.close()

    print(f"Saved confusion matrix to {output_dir}/confusion_matrix_multiclass.png")

def main():
    parser = argparse.ArgumentParser(description="Train multiclass CNN")
    parser.add_argument("--train-csv", required=True, help="Training CSV file")
    parser.add_argument("--val-csv", required=True, help="Validation CSV file")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--max-length", type=int, default=116, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    print(f"Train: {len(train_df)} sequences")
    print(f"Val:   {len(val_df)} sequences")

    print("\nTrain class distribution:")
    print(train_df['class_name'].value_counts())
    print("\nVal class distribution:")
    print(val_df['class_name'].value_counts())

    # Prepare tensors
    print("\nPreparing tensors...")
    train_X = sequences_to_tensor(train_df['sequence'].tolist(), args.max_length)
    train_y = torch.tensor(train_df['label'].values, dtype=torch.long)

    val_X = sequences_to_tensor(val_df['sequence'].tolist(), args.max_length)
    val_y = torch.tensor(val_df['label'].values, dtype=torch.long)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        TensorDataset(val_X, val_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Model
    print("\nInitializing model...")
    model = VGeneCNN(
        input_channels=20,
        seq_length=args.max_length,
        num_filters=[64, 128, 256],
        kernel_size=3,
        dropout=0.3
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # For multiclass
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    best_f1 = 0.0

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, 'best_model_multiclass.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  💾 Best model saved (F1: {best_f1:.4f})")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Load best model
    model.load_state_dict(torch.load(model_path))
    val_metrics = evaluate(model, val_loader, criterion, device)

    print(f"\nBest validation F1: {best_f1:.4f}")
    print(f"Final validation accuracy: {val_metrics['accuracy']:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_metrics['labels'], val_metrics['predictions'],
                                target_names=CLASS_NAMES, digits=4))

    # Save results
    os.makedirs('results', exist_ok=True)

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv('results/training_history_multiclass.csv', index=False)
    print("\nSaved training history to results/training_history_multiclass.csv")

    # Plot training curves
    plot_training_history(history, 'results')

    # Plot confusion matrix
    plot_confusion_matrix(val_metrics['labels'], val_metrics['predictions'], 'results')

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model: {model_path}")
    print(f"Best F1 score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
