"""
Sequence encoding functions
"""

import numpy as np
import torch

# 20 standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def one_hot_encode(sequence, max_length=116):
    """
    One-hot encode a protein sequence

    Args:
        sequence: String of amino acids
        max_length: Maximum sequence length (padding)

    Returns:
        numpy array of shape (max_length, 20)
    """
    # Initialize with zeros
    encoding = np.zeros((max_length, 20), dtype=np.float32)

    # Encode each amino acid
    for i, aa in enumerate(sequence[:max_length]):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1.0

    return encoding


def encode_sequences(sequences, max_length=116):
    """
    Batch encode multiple sequences

    Args:
        sequences: List of amino acid sequences
        max_length: Maximum sequence length

    Returns:
        numpy array of shape (num_sequences, max_length, 20)
    """
    encoded = np.array([one_hot_encode(seq, max_length) for seq in sequences])
    return encoded


def sequences_to_tensor(sequences, max_length=116):
    """
    Convert sequences to PyTorch tensor

    Args:
        sequences: List of amino acid sequences
        max_length: Maximum sequence length

    Returns:
        PyTorch tensor of shape (num_sequences, 20, max_length)
        Note: Channels first for Conv1d
    """
    encoded = encode_sequences(sequences, max_length)
    # Transpose to (batch, channels, length) for Conv1d
    tensor = torch.from_numpy(encoded).permute(0, 2, 1)
    return tensor
