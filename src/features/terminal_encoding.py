"""
Olivieri Encoding Method

Based on Olivieri's original code for V-gene classification.
This encoding achieved >90% precision on IMGT validation.

Key insight: Instead of using full sequence one-hot encoding,
use one-hot of terminal regions + dipeptide counts.

Author: Luis Rana Cortizo (implementing Olivieri's method)
Date: 2026-02-06
"""

import numpy as np
from typing import List, Dict
from collections import Counter


# Standard amino acid alphabet (same as Olivieri)
AA_ALPHABET = 'ARNDCQEGHILKMFPSTWYV'  # Note: Olivieri's order
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


class TerminalRegionEncoder:
    """
    Olivieri's encoding method for V-gene classification.
    
    Features:
    1. One-hot encoding of first 40 + last 40 amino acids (1,600 features)
    2. Dipeptide counts from full sequence (400 features)
    
    Total: 2,000 fixed-length features regardless of sequence length
    
    Biological rationale:
    - N-terminal 40 aa: Contains Framework 1 (FR1) - highly conserved
    - C-terminal 40 aa: Contains conserved cysteine and FR3
    - Dipeptide counts: Capture global sequence composition
    """
    
    def __init__(self):
        self.n_features = 2000
        self.n_terminal = 40
        self.c_terminal = 40
        
        # Pre-compute all dipeptides
        self.dipeptides = []
        for aa1 in AA_ALPHABET:
            for aa2 in AA_ALPHABET:
                self.dipeptides.append(aa1 + aa2)
        
        assert len(self.dipeptides) == 400, "Should have 400 dipeptides"
    
    def _onehot_extremes(self, sequence: str) -> np.ndarray:
        """
        One-hot encode first 40 + last 40 amino acids.
        
        Returns:
            np.ndarray of shape (1600,) - flattened one-hot
        """
        # Extract extremes
        n_term = sequence[:self.n_terminal]
        c_term = sequence[-self.c_terminal:]
        extremes = n_term + c_term
        
        # Pad if too short
        if len(extremes) < 80:
            extremes = extremes + 'X' * (80 - len(extremes))
        
        # One-hot encode
        onehot = []
        for aa in extremes:
            for ref_aa in AA_ALPHABET:
                if aa == ref_aa:
                    onehot.append(1)
                else:
                    onehot.append(0)
        
        return np.array(onehot, dtype=np.int32)
    
    def _dipeptide_counts(self, sequence: str) -> np.ndarray:
        """
        Count all dipeptides in full sequence.
        
        Returns:
            np.ndarray of shape (400,) - counts for each dipeptide
        """
        counts = np.zeros(400, dtype=np.int32)
        
        for i, dipep in enumerate(self.dipeptides):
            count = sequence.count(dipep)
            counts[i] = count
        
        return counts
    
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode a single sequence using Olivieri's method.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            np.ndarray of shape (2000,)
        """
        # Part 1: One-hot of extremes (1600 features)
        onehot = self._onehot_extremes(sequence)
        
        # Part 2: Dipeptide counts (400 features)
        dipep = self._dipeptide_counts(sequence)
        
        # Concatenate
        encoding = np.concatenate([onehot, dipep])
        
        assert encoding.shape == (2000,), f"Expected shape (2000,), got {encoding.shape}"
        
        return encoding.astype(np.float32)
    
    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Encode a batch of sequences.
        
        Returns:
            np.ndarray of shape (n_sequences, 2000)
        """
        encodings = []
        for seq in sequences:
            enc = self.encode(seq)
            encodings.append(enc)
        
        return np.array(encodings, dtype=np.float32)


class TerminalRegionEncoderNormalized(TerminalRegionEncoder):
    """
    Olivieri encoder with optional normalization.
    
    Adds:
    - Normalize dipeptide counts by sequence length
    - Optional L2 normalization of full feature vector
    """
    
    def __init__(self, normalize_dipeptides: bool = True, l2_normalize: bool = False):
        super().__init__()
        self.normalize_dipeptides = normalize_dipeptides
        self.l2_normalize = l2_normalize
    
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode with optional normalization.
        """
        # Part 1: One-hot (unchanged)
        onehot = self._onehot_extremes(sequence)
        
        # Part 2: Dipeptide counts
        dipep = self._dipeptide_counts(sequence)
        
        # Normalize dipeptide counts by sequence length
        if self.normalize_dipeptides:
            seq_length = len(sequence)
            if seq_length > 1:  # Avoid division by zero
                dipep = dipep.astype(np.float32) / (seq_length - 1)  # -1 because dipeptides
        
        # Concatenate
        encoding = np.concatenate([onehot, dipep])
        
        # Optional L2 normalization
        if self.l2_normalize:
            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm
        
        return encoding.astype(np.float32)


def compare_encodings(sequence: str):
    """
    Compare original vs normalized Olivieri encoding on a sample sequence.
    """
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}\n")
    
    # Original
    enc_original = TerminalRegionEncoder()
    feat_original = enc_original.encode(sequence)
    
    print("Original Olivieri Encoding:")
    print(f"  Shape: {feat_original.shape}")
    print(f"  One-hot part (first 1600): sum = {feat_original[:1600].sum()}")
    print(f"  Dipeptide part (last 400): sum = {feat_original[1600:].sum()}")
    print(f"  Dipeptide part (last 400): max = {feat_original[1600:].max()}")
    print()
    
    # Normalized dipeptides
    enc_norm = TerminalRegionEncoderNormalized(normalize_dipeptides=True, l2_normalize=False)
    feat_norm = enc_norm.encode(sequence)
    
    print("Olivieri + Normalized Dipeptides:")
    print(f"  Shape: {feat_norm.shape}")
    print(f"  One-hot part (first 1600): sum = {feat_norm[:1600].sum()}")
    print(f"  Dipeptide part (last 400): sum = {feat_norm[1600:].sum():.4f}")
    print(f"  Dipeptide part (last 400): max = {feat_norm[1600:].max():.4f}")
    print()
    
    # Full L2 normalization
    enc_l2 = TerminalRegionEncoderNormalized(normalize_dipeptides=True, l2_normalize=True)
    feat_l2 = enc_l2.encode(sequence)
    
    print("Olivieri + L2 Normalized:")
    print(f"  Shape: {feat_l2.shape}")
    print(f"  L2 norm: {np.linalg.norm(feat_l2):.4f}")
    print()


# Utility function to get encoder by name
def get_olivieri_encoder(normalize_dipeptides: bool = False, 
                         l2_normalize: bool = False) -> TerminalRegionEncoder:
    """
    Factory function to get Olivieri encoder.
    
    Args:
        normalize_dipeptides: If True, normalize dipeptide counts by sequence length
        l2_normalize: If True, apply L2 normalization to full feature vector
    
    Returns:
        TerminalRegionEncoder instance
    """
    if normalize_dipeptides or l2_normalize:
        return TerminalRegionEncoderNormalized(normalize_dipeptides, l2_normalize)
    else:
        return TerminalRegionEncoder()


if __name__ == '__main__':
    # Test with sample V-gene sequence
    test_seq = ("QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGT"
                "NYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARYDYYGSSYFDYWGQGTLVTVSS")
    
    print("="*80)
    print("TESTING OLIVIERI ENCODING METHOD")
    print("="*80)
    print()
    
    compare_encodings(test_seq)
    
    print("="*80)
    print("ENCODING VALIDATION")
    print("="*80)
    
    # Test that it works with sequences of different lengths
    sequences = [
        "QVQLVQSGAEVKKPGASVKVSCKAS",  # Short (25 aa)
        test_seq,                      # Medium (116 aa)
        test_seq * 2                   # Long (232 aa)
    ]
    
    encoder = TerminalRegionEncoder()
    
    for i, seq in enumerate(sequences):
        enc = encoder.encode(seq)
        print(f"Sequence {i+1} (length={len(seq)}): encoding shape = {enc.shape}")
    
    print()
    print("✅ All sequences encoded to fixed-length vector (2000,)")
    print()
    
    # Batch encoding test
    batch_enc = encoder.encode_batch(sequences)
    print(f"Batch encoding shape: {batch_enc.shape}")
    print(f"Expected: ({len(sequences)}, 2000)")
