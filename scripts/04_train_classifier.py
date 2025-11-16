"""
Train V-gene classifier using CNN
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path before other imports so local package can be imported cleanly
sys.path.append(str(Path(__file__).parent.parent))

from src.features.encoding import sequences_to_tensor  # noqa: E402
from src.models.classifier import VGeneCNN  # noqa: E402
from src.models.train import train_epoch, evaluate  # noqa: E402

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# Custom Dataset
class ProteinDataset(Dataset):
    """Dataset for protein sequences"""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Load data
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Encode sequences
print("\nEncoding sequences...")
X_train = sequences_to_tensor(train_df["sequence"].tolist())
y_train = torch.tensor(train_df["label"].values, dtype=torch.float32).unsqueeze(1)

X_val = sequences_to_tensor(val_df["sequence"].tolist())
y_val = torch.tensor(val_df["label"].values, dtype=torch.float32).unsqueeze(1)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Create datasets and dataloaders
train_dataset = ProteinDataset(X_train, y_train)
val_dataset = ProteinDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
print("\n" + "=" * 70)
print("INITIALIZING MODEL")
print("=" * 70)

model = VGeneCNN(
    input_channels=20,
    seq_length=116,
    num_filters=[64, 128, 256],
    kernel_size=3,
    dropout=0.3,
)
model = model.to(DEVICE)

print(f"Total parameters: {model.count_parameters():,}")

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": [],
    "val_auc": [],
}

best_val_f1 = 0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    # Evaluate
    val_metrics = evaluate(model, val_loader, criterion, DEVICE)

    # Store history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_metrics["loss"])
    history["val_accuracy"].append(val_metrics["accuracy"])
    history["val_precision"].append(val_metrics["precision"])
    history["val_recall"].append(val_metrics["recall"])
    history["val_f1"].append(val_metrics["f1"])
    history["val_auc"].append(val_metrics["auc"])

    # Print progress
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(
        f"  Val Loss: {val_metrics['loss']:.4f} | "
        f"Acc: {val_metrics['accuracy']:.4f} | "
        f"F1: {val_metrics['f1']:.4f} | "
        f"AUC: {val_metrics['auc']:.4f}"
    )

    # Save best model
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch + 1
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
        print(f"  ✅ New best model saved (F1: {best_val_f1:.4f})")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best F1 score: {best_val_f1:.4f} at epoch {best_epoch}")

# Plot training history
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
axes[0, 0].plot(history["train_loss"], label="Train")
axes[0, 0].plot(history["val_loss"], label="Validation")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training and Validation Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy
axes[0, 1].plot(history["val_accuracy"])
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Validation Accuracy")
axes[0, 1].grid(True)

# F1 Score
axes[1, 0].plot(history["val_f1"])
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("F1 Score")
axes[1, 0].set_title("Validation F1 Score")
axes[1, 0].grid(True)

# AUC
axes[1, 1].plot(history["val_auc"])
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("AUC-ROC")
axes[1, 1].set_title("Validation AUC-ROC")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_history.png", dpi=300, bbox_inches="tight")
print(f"\n📊 Training plots saved to {RESULTS_DIR / 'training_history.png'}")

# Save history to CSV
history_df = pd.DataFrame(history)
history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
print(f"📊 Training history saved to {RESULTS_DIR / 'training_history.csv'}")

print("\n✅ Done!")
