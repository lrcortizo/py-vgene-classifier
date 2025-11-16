"""
CNN classifier for V-gene detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGeneCNN(nn.Module):
    """
    1D Convolutional Neural Network for V-gene classification

    Architecture:
    - Conv1D layers to detect sequence motifs
    - MaxPooling for translation invariance
    - Fully connected layers for classification
    """

    def __init__(
        self,
        input_channels=20,
        seq_length=116,
        num_filters=[64, 128, 256],
        kernel_size=3,
        dropout=0.3,
    ):
        """
        Args:
            input_channels: Number of input channels (20 for one-hot amino acids)
            seq_length: Length of input sequences
            num_filters: List of filter numbers for each conv layer
            kernel_size: Size of convolutional kernels
            dropout: Dropout probability
        """
        super(VGeneCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_filters[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' padding
        )
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=num_filters[0],
            out_channels=num_filters[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=num_filters[1],
            out_channels=num_filters[2],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn3 = nn.BatchNorm1d(num_filters[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Calculate size after convolutions and pooling
        # seq_length / 2 / 2 / 2 = seq_length / 8
        pooled_length = seq_length // 8
        fc_input_size = num_filters[2] * pooled_length

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)  # Binary classification

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, channels=20, length=116)

        Returns:
            Output tensor of shape (batch, 1) with sigmoid activation
        """
        # Convolutional blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        # Sigmoid for binary classification
        x = torch.sigmoid(x)

        return x

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
