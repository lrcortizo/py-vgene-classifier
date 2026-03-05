"""
Training and evaluation functions
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set

    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()

            # Collect predictions
            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            labels = batch_y.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": (
            roc_auc_score(all_labels, all_probs)
            if len(np.unique(all_labels)) > 1
            else 0.0
        ),
    }

    return metrics
