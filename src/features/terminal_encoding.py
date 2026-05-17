"""
Olivieri Encoding Method

Based on Olivieri's original code for V-gene classification.
This encoding achieved >90% precision on IMGT validation.

Key insight: Instead of using full sequence one-hot encoding,
use one-hot of terminal regions + dipeptide counts.

Author: Luis Rana Cortizo (implementing Olivieri's method)
Date: 2026-02-06
"""

import re
import numpy as np
from typing import List, Dict
from collections import Counter


# Standard amino acid alphabet (same as Olivieri)
AA_ALPHABET = 'ARNDCQEGHILKMFPSTWYV'  # Note: Olivieri's order
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Names of the 5 conserved V-gene motif flags (used by TerminalRegionEncoderV3)
MOTIF_FLAG_NAMES = ['C23', 'W41', 'YYC', 'C104', 'WGXG']

# ── AAindex1 physicochemical scales (used by TerminalRegionEncoderV3) ────────
# All arrays follow AA_ALPHABET order: 'ARNDCQEGHILKMFPSTWYV'
PHYSICOCHEMICAL = {
    'hydrophobicity': np.array([          # Kyte-Doolittle 1982
         1.8, -4.5, -3.5, -3.5,  2.5,
        -3.5, -3.5, -0.4, -3.2,  4.5,
         3.8, -3.9,  1.9,  2.8, -1.6,
        -0.8, -0.7, -0.9, -1.3,  4.2], dtype=np.float32),
    'volume': np.array([                  # Pontius 1996 (Å³)
        67.0, 148.0,  96.0,  91.0,  86.0,
       114.0, 109.0,  48.0, 118.0, 124.0,
       124.0, 135.0, 124.0, 135.0,  90.0,
        73.0,  93.0, 163.0, 141.0, 105.0], dtype=np.float32),
    'charge': np.array([                  # Net charge at pH 7 (approximate)
         0.0,  1.0,  0.0, -1.0,  0.0,
         0.0, -1.0,  0.0,  0.0,  0.0,
         0.0,  1.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),
    'flexibility': np.array([             # Vihinen 1994 (normalized B-factor)
        0.357, 0.529, 0.463, 0.511, 0.346,
        0.493, 0.497, 0.544, 0.323, 0.462,
        0.365, 0.466, 0.295, 0.314, 0.509,
        0.507, 0.444, 0.305, 0.420, 0.386], dtype=np.float32),
    'polarity': np.array([                # Zimmerman 1968
         0.00, 52.00,  3.38, 49.70,  1.48,
         3.53, 49.90,  0.00, 51.60,  0.13,
         0.13, 49.50,  1.43,  0.35,  1.58,
         1.67,  1.66,  2.10,  1.61,  0.13], dtype=np.float32),
}
_PROP_NAMES = list(PHYSICOCHEMICAL.keys())  # fixed iteration order


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


class TerminalRegionEncoderV3(TerminalRegionEncoder):
    """
    V3 encoding: extends TerminalRegionEncoder with 45 additional features.

    Feature vector layout (2045 dims):
      [0:800]     One-hot N-terminal 40 aa          (from parent)
      [800:1600]  One-hot C-terminal 40 aa          (from parent)
      [1600:2000] Dipeptide counts (full sequence)  (from parent)
      [2000:2020] AA frequency histogram            (NEW — 20 dims)
      [2020:2040] Physicochemical properties        (NEW — 20 dims)
      [2040:2045] Conserved motif flags             (NEW —  5 dims)

    Physicochemical block (20 dims):
      5 properties x (mean, std) x (N-term 40aa, C-term 40aa)
      Properties: hydrophobicity, volume, charge, flexibility, polarity

    Conserved motif flags (5 dims, all binary):
      0: first C in sequence[:30]             (C23 proxy, FR1)
      1: W in sequence[35:45]                (W41 proxy, FR2)
      2: 'YYC' in sequence[-25:]             (FR3 conserved triad)
      3: C in sequence[-20:]                 (C104 proxy, FR3)
      4: re.search(r'WG.G', sequence[-25:])  (WGXG motif, FR3-J junction)

    Requires retraining: CNN input_size must be updated from 2000 to 2045.
    Original TerminalRegionEncoder (2000 dims) is unchanged.
    """

    def __init__(self):
        super().__init__()
        self.n_features = 2045

    # ── New feature methods ──────────────────────────────────────────────────

    def _aa_frequency(self, sequence: str) -> np.ndarray:
        """Normalized frequency of each of 20 standard AA over full sequence."""
        total = len(sequence) or 1
        counts = Counter(sequence)
        freq = np.array(
            [counts.get(aa, 0) / total for aa in AA_ALPHABET],
            dtype=np.float32
        )
        return freq  # (20,)

    def _physicochemical(self, sequence: str) -> np.ndarray:
        """
        For each of 5 properties: (mean, std) computed separately on
        N-terminal 40 aa and C-terminal 40 aa.
        5 props x 2 stats x 2 regions = 20 dims.
        """
        n_term = sequence[:self.n_terminal]
        c_term = sequence[-self.c_terminal:]
        result = []
        for prop_name in _PROP_NAMES:
            scale = PHYSICOCHEMICAL[prop_name]
            for region in (n_term, c_term):
                indices = [AA_TO_IDX[aa] for aa in region if aa in AA_TO_IDX]
                if indices:
                    vals = scale[np.array(indices)]
                    result.append(float(vals.mean()))
                    result.append(float(vals.std()))
                else:
                    result.extend([0.0, 0.0])
        return np.array(result, dtype=np.float32)  # (20,)

    def _motif_flags(self, sequence: str) -> np.ndarray:
        """
        5 binary flags for structurally conserved V-gene positions/motifs.
        Uses relative search windows instead of fixed IMGT positions —
        robust to FR1 length variation across species.
        """
        flags = np.zeros(5, dtype=np.float32)

        # 0: C23 proxy — first Cys in N-terminal 30 aa (FR1)
        flags[0] = 1.0 if 'C' in sequence[:30] else 0.0

        # 1: W41 proxy — Trp anywhere in positions 35-45 (FR2 entry)
        flags[1] = 1.0 if 'W' in sequence[35:45] else 0.0

        # 2: YYC motif — conserved triad in C-terminal 25 aa (FR3)
        flags[2] = 1.0 if 'YYC' in sequence[-25:] else 0.0

        # 3: C104 proxy — any Cys in last 20 aa (FR3)
        flags[3] = 1.0 if 'C' in sequence[-20:] else 0.0

        # 4: WGXG motif — FR3-J junction signature
        flags[4] = 1.0 if re.search(r'WG.G', sequence[-25:]) else 0.0

        return flags  # (5,)

    # ── Override encode ──────────────────────────────────────────────────────

    def encode_with_flags(self, sequence: str):
        """
        Like encode(), but also returns the 5 motif flags as a named dict.

        Returns:
            encoding : np.ndarray of shape (2045,), dtype float32
            flag_dict: dict mapping MOTIF_FLAG_NAMES → bool
        """
        encoding = self.encode(sequence)
        raw_flags = self._motif_flags(sequence)
        flag_dict = {
            name: bool(raw_flags[i])
            for i, name in enumerate(MOTIF_FLAG_NAMES)
        }
        return encoding, flag_dict

    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode a single sequence using V3 method.

        Returns:
            np.ndarray of shape (2045,), dtype float32
        """
        onehot  = self._onehot_extremes(sequence)   # (1600,) from parent
        dipep   = self._dipeptide_counts(sequence)  # (400,)  from parent
        aa_freq = self._aa_frequency(sequence)      # (20,)   NEW
        physico = self._physicochemical(sequence)   # (20,)   NEW
        motifs  = self._motif_flags(sequence)       # (5,)    NEW

        encoding = np.concatenate([onehot, dipep, aa_freq, physico, motifs])
        assert encoding.shape == (2045,), \
            f"V3 encoder: expected (2045,), got {encoding.shape}"
        return encoding.astype(np.float32)


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
    print("OK: All sequences encoded to fixed-length vector (2000,)")
    print()
    
    # Batch encoding test
    batch_enc = encoder.encode_batch(sequences)
    print(f"Batch encoding shape: {batch_enc.shape}")
    print(f"Expected: ({len(sequences)}, 2000)")

    print()
    print("=" * 80)
    print("TESTING V3 ENCODER")
    print("=" * 80)

    enc_v3 = TerminalRegionEncoderV3()
    feat_v3 = enc_v3.encode(test_seq)

    print(f"V3 encoding shape:             {feat_v3.shape}   (expected: (2045,))")
    print(f"  [0:1600]    one-hot:         sum={feat_v3[:1600].sum():.0f}")
    print(f"  [1600:2000] dipeptides:      sum={feat_v3[1600:2000].sum():.0f}")
    print(f"  [2000:2020] aa_freq:         sum={feat_v3[2000:2020].sum():.4f}  (should be ~1.0)")
    print(f"  [2020:2040] physicochemical: {feat_v3[2020:2040].round(3)}")
    print(f"  [2040:2045] motif flags:     {feat_v3[2040:2045]}")
    assert feat_v3.shape == (2045,), f"FAIL: shape={feat_v3.shape}"
    assert feat_v3.dtype == np.float32, f"FAIL: dtype={feat_v3.dtype}"
    print()
    print("OK: TerminalRegionEncoderV3 passed all checks")
