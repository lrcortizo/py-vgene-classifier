#!/usr/bin/env python3
"""
Generate synthetic background sequences.

Creates random protein sequences to use as negative examples
for V-gene classification training.

As recommended by supervisor: for initial training phase, synthetic
background is sufficient ("secuencias que claramente no sean Vs").

Usage:
    python scripts/05b_generate_synthetic_background.py
"""

import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path


def generate_synthetic_background(
    num_sequences: int = 37814,
    min_length: int = 85,
    max_length: int = 105,
    seed: int = 42
):
    """
    Generate random protein sequences.

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length (amino acids)
        max_length: Maximum sequence length (amino acids)
        seed: Random seed for reproducibility

    Returns:
        List of SeqRecord objects
    """
    # Standard 20 amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # Set seed for reproducibility
    random.seed(seed)

    records = []

    for i in range(num_sequences):
        # Random length
        length = random.randint(min_length, max_length)

        # Random sequence
        sequence = ''.join(random.choices(amino_acids, k=length))

        # Create record
        rec = SeqRecord(
            Seq(sequence),
            id=f"synthetic_bg_{i+1}",
            description="synthetic_background"
        )
        records.append(rec)

        # Progress
        if (i + 1) % 5000 == 0:
            print(f"   {i+1}/{num_sequences}...")

    return records


def main():
    print("=" * 70)
    print("GENERATE SYNTHETIC BACKGROUND SEQUENCES")
    print("=" * 70)

    # Parameters
    num_sequences = 37814
    min_length = 85
    max_length = 105
    seed = 42

    print(f"\n📋 Parameters:")
    print(f"   Number of sequences: {num_sequences}")
    print(f"   Length range: {min_length}-{max_length} aa")
    print(f"   Random seed: {seed}")

    # Generate
    print(f"\n🎲 Generating random sequences...")
    records = generate_synthetic_background(
        num_sequences=num_sequences,
        min_length=min_length,
        max_length=max_length,
        seed=seed
    )

    # Save
    output_dir = Path("data/raw/negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "background_synthetic.fasta"

    print(f"\n💾 Saving to {output_file}...")
    SeqIO.write(records, output_file, "fasta")

    # Summary
    lengths = [len(rec.seq) for rec in records]
    print(f"\n✅ DONE")
    print(f"\n📊 Summary:")
    print(f"   Total sequences: {len(records)}")
    print(f"   Length range: {min(lengths)}-{max(lengths)} aa")
    print(f"   Mean length: {sum(lengths)/len(lengths):.1f} aa")
    print(f"   Output file: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 70)
    print("Next step:")
    print("  python scripts/05_prepare_multispecies_dataset.py \\")
    print("      --input-dir /path/to/alignment_to_imgt \\")
    print("      --loci ighv igkv trav trbv \\")
    print("      --output-dir data/processed")
    print("=" * 70)


if __name__ == "__main__":
    main()
