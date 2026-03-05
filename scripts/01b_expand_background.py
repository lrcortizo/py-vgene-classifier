#!/usr/bin/env python3
"""
Expand hard negative sequences via conservative mutations.
Creates realistic variants while maintaining protein structure.
"""

import argparse
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path

# Conservative amino acid substitutions (BLOSUM groups)
CONSERVATIVE_MUTATIONS = {
    'A': ['S', 'T'],
    'S': ['A', 'T'],
    'T': ['A', 'S'],
    'V': ['I', 'L'],
    'I': ['V', 'L'],
    'L': ['I', 'V'],
    'D': ['E', 'N'],
    'E': ['D', 'Q'],
    'N': ['D', 'S'],
    'Q': ['E', 'K'],
    'K': ['R', 'Q'],
    'R': ['K'],
    'F': ['Y', 'W'],
    'Y': ['F', 'W'],
    'W': ['F', 'Y'],
    'H': ['N', 'Q'],
    'P': ['P'],  # Proline rarely substitutes
    'G': ['A'],  # Glycine mostly to Ala
    'C': ['S'],  # Cysteine rarely changes
    'M': ['L', 'I'],
}


def mutate_sequence(seq_str, mutation_rate=0.02, seed=None):
    """
    Apply conservative mutations to a sequence.

    Args:
        seq_str: Original sequence
        mutation_rate: Fraction of positions to mutate (default: 2%)
        seed: Random seed for reproducibility

    Returns:
        Mutated sequence string
    """
    if seed is not None:
        random.seed(seed)

    seq_list = list(seq_str)
    n_mutations = max(1, int(len(seq_str) * mutation_rate))

    # Select random positions
    positions = random.sample(range(len(seq_str)), n_mutations)

    for pos in positions:
        original = seq_str[pos]
        if original in CONSERVATIVE_MUTATIONS:
            # Conservative substitution
            options = CONSERVATIVE_MUTATIONS[original]
            if options:
                seq_list[pos] = random.choice(options)

    return ''.join(seq_list)


def generate_variants(
    input_fasta,
    output_fasta,
    target_total=8000,
    mutation_rate=0.02,
    seed=42
):
    """
    Generate variants of input sequences.

    Args:
        input_fasta: Input FASTA with hard negatives
        output_fasta: Output FASTA with expanded set
        target_total: Target number of sequences
        mutation_rate: Mutation rate per variant
        seed: Random seed
    """
    random.seed(seed)

    # Load original sequences
    originals = list(SeqIO.parse(input_fasta, 'fasta'))

    print(f"Original sequences: {len(originals)}")
    print(f"Target total: {target_total}")

    # Calculate how many variants per original
    n_variants_per_seq = (target_total // len(originals)) + 1

    print(f"Generating {n_variants_per_seq} variants per sequence...")

    all_sequences = []

    # Keep originals
    for rec in originals:
        new_rec = SeqRecord(
            rec.seq,
            id=f"{rec.id}_original",
            description=rec.description
        )
        all_sequences.append(new_rec)

    # Generate variants
    for i, rec in enumerate(originals):
        seq_str = str(rec.seq)

        for variant_num in range(n_variants_per_seq):
            # Apply mutations with different seeds
            variant_seed = seed + i * 1000 + variant_num
            mutated = mutate_sequence(seq_str, mutation_rate, variant_seed)

            new_rec = SeqRecord(
                Seq(mutated),
                id=f"{rec.id}_v{variant_num+1}",
                description=f"variant of {rec.id} (mut_rate={mutation_rate})"
            )
            all_sequences.append(new_rec)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(originals)}... ({len(all_sequences)} total)")

    # Trim to target
    if len(all_sequences) > target_total:
        random.shuffle(all_sequences)
        all_sequences = all_sequences[:target_total]

    # Save
    SeqIO.write(all_sequences, output_fasta, 'fasta')

    print(f"\n✅ Generated {len(all_sequences)} sequences")
    print(f"   Saved to: {output_fasta}")

    return len(all_sequences)


def main():
    parser = argparse.ArgumentParser(
        description="Expand hard negatives via conservative mutations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input FASTA with hard negatives"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output FASTA with expanded set"
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=8000,
        help="Target number of sequences (default: 8000)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.02,
        help="Mutation rate per variant (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("EXPAND HARD NEGATIVES VIA DATA AUGMENTATION")
    print("=" * 80)
    print(f"Strategy: Conservative amino acid substitutions")
    print(f"Mutation rate: {args.mutation_rate * 100:.1f}% per variant")
    print("=" * 80)
    print()

    generate_variants(
        args.input,
        args.output,
        args.target_total,
        args.mutation_rate,
        args.seed
    )

    print()
    print("=" * 80)
    print("✅ DATA AUGMENTATION COMPLETE")
    print("=" * 80)
    print()
    print("Next: Combine with V-genes to create training dataset")


if __name__ == "__main__":
    main()
