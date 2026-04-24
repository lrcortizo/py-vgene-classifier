"""
CNN model for terminal-region encoding
"""

import torch.nn as nn


class CNN_TerminalEncoding(nn.Module):
    """
    CNN adapted for fixed-length feature vectors (terminal-region encoding).
    
    Treats the 2000 features (N-term + C-term + dipeptides) as a 1D sequence:
    - Reshape to (batch, 1, 2000)
    - Apply 1D convolutions
    - Fully connected layers
    """
    
    def __init__(self, input_size=2000, num_classes=5):
        super().__init__()
        
        # Treat 2000 features as sequence of length 2000 with 1 channel
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Dynamic flatten size: 3x MaxPool1d(2) halves the dimension each time
        # input_size=2000 -> 1000 -> 500 -> 250 -> flatten=64000 (unchanged)
        # input_size=2045 -> 1022 -> 511 -> 255 -> flatten=65280 (V3 encoder)
        _s = input_size
        for _ in range(3):
            _s = _s // 2
        flatten_size = 256 * _s
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 2000)
        # Reshape to (batch, 1, 2000) for Conv1d
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x
