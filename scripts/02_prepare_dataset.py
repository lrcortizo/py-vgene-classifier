#!/usr/bin/env python3
"""
Script 02: Prepare Multi-Species Multiclass Dataset (Simplified for raw FASTAs)
Version: 2.0.0
Purpose: Prepare training data from simple IMGT FASTA files

CLASSES (5 total):
  0 = background (hard negatives: MHC, C-regions, Ig-superfamily)
  1 = IGHV (IG Heavy Variable)
  2 = IGKV (IG Kappa Light Variable)
  3 = TRAV (TCR Alpha Variable)
  4 = TRBV (TCR Beta Variable)

USAGE:
    python scripts/02_prepare_dataset.py \
        --input-dir data/raw/positive \
        --background data/background/background_hard_negatives.fasta \
        --output-dir data/processed

"""

import os
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import random

# Map loci to class labels
LOCUS_TO_CLASS = {
    'ighv': 1,
    'igkv': 2,
    'iglv': 2,  # IGLV grouped with IGKV (light chain)
    'trav': 3,
    'trbv': 4,
    'trdv': 4,  # TRDV grouped with TRBV (delta chain)
    'trgv': 3,  # TRGV grouped with TRAV (gamma chain)
    'background': 0,
}

CLASS_NAMES = {
    0: 'background',
    1: 'IGHV',
    2: 'IGKV',
    3: 'TRAV',
    4: 'TRBV'
}


def parse_simple_header(header):
    """
    Parse simple IMGT header.

    Format: > Homsap_IGHV1-2 IGHV1-2*01_X07448_VH_F

    Returns:
        Dict with id, gene_name, species, locus
    """
    try:
        parts = header.strip().split()
        if len(parts) < 2:
            return None

        # First part: Species_Gene (e.g., Homsap_IGHV1-2)
        species_gene = parts[0].replace('>', '').strip()
        if '_' in species_gene:
            species, gene = species_gene.split('_', 1)
        else:
            species = 'unknown'
            gene = species_gene

        # Extract locus from gene name (first 4-5 chars)
        locus = None
        for locus_key in LOCUS_TO_CLASS.keys():
            if locus_key != 'background' and gene.lower().startswith(locus_key):
                locus = locus_key
                break

        if not locus:
            return None

        # Second part: Full gene name
        full_name = parts[1] if len(parts) > 1 else gene

        info = {
            "id": full_name,
            "gene_name": gene,
            "species": species,
            "locus": locus
        }
        return info

    except Exception as e:
        print(f"⚠️  Could not parse header: {header[:50]}... Error: {e}")
        return None


def clean_sequence(seq_str):
    """
    Remove alignment gaps and non-alphabetic characters.

    IMGT alignments use:
      . = gap in numbering
      - = deletion
    """
    # Remove gaps and dots
    cleaned = seq_str.replace("-", "").replace(".", "")
    # Keep only alphabetic characters
    cleaned = "".join(c for c in cleaned if c.isalpha())
    return cleaned.upper()


def load_vgenes_simple(input_dir, loci, min_length=80, max_length=140):
    """
    Load V-genes from simple IMGT FASTA files.

    Expected files: ighv.fasta, igkv.fasta, trav.fasta, trbv.fasta

    Args:
        input_dir: Directory containing FASTA files
        loci: List of loci to load
        min_length: Minimum sequence length (for terminal-region encoding)
        max_length: Maximum sequence length

    Returns:
        List of dicts with sequence records
    """
    vgene_records = []
    stats = {locus: 0 for locus in loci}

    for locus in loci:
        fasta_file = Path(input_dir) / f"{locus}.fasta"

        if not fasta_file.exists():
            print(f"⚠️  File not found: {fasta_file}")
            continue

        print(f"   Loading {fasta_file.name}...")

        for record in SeqIO.parse(fasta_file, "fasta"):
            # Parse header
            header_info = parse_simple_header(record.description)
            if not header_info:
                continue

            # Clean sequence (remove gaps)
            clean_seq = clean_sequence(str(record.seq))

            # Filter by length
            if len(clean_seq) < min_length or len(clean_seq) > max_length:
                continue

            # Get class label
            class_label = LOCUS_TO_CLASS[locus]

            vgene_records.append({
                "id": header_info["id"],
                "sequence": clean_seq,
                "length": len(clean_seq),
                "species": header_info["species"],
                "locus": locus,
                "label": class_label,
                "class_name": CLASS_NAMES[class_label]
            })

            stats[locus] += 1

    print(f"\n   V-genes loaded:")
    for locus, count in stats.items():
        if count > 0:
            print(f"     {locus.upper()}: {count:,}")
    print(f"     Total: {len(vgene_records):,}")

    return vgene_records


def load_background(background_file, num_needed, min_length=80, max_length=140, seed=42):
    """Load background sequences (hard negatives)."""
    if not os.path.exists(background_file):
        print(f"❌ Background file not found: {background_file}")
        return []

    print(f"   Loading background from {Path(background_file).name}...")
    all_background = list(SeqIO.parse(background_file, "fasta"))

    # Filter by length
    filtered = [rec for rec in all_background
                if min_length <= len(rec.seq) <= max_length]

    print(f"     Total available: {len(all_background):,}")
    print(f"     After length filter ({min_length}-{max_length} aa): {len(filtered):,}")

    # Sample if needed
    random.seed(seed)
    if len(filtered) > num_needed:
        sampled = random.sample(filtered, num_needed)
        print(f"     Sampled: {len(sampled):,}")
    else:
        sampled = filtered
        if len(filtered) < num_needed:
            print(f"     ⚠️  Only {len(filtered):,} available (target: {num_needed:,})")

    background_records = []
    for rec in sampled:
        background_records.append({
            "id": rec.id,
            "sequence": str(rec.seq).upper(),
            "length": len(rec.seq),
            "species": "background",
            "locus": "background",
            "label": 0,
            "class_name": "background"
        })

    print(f"     Background sequences loaded: {len(background_records):,}")

    return background_records


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multiclass dataset for V-gene classification (v2.0.0)"
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/positive",
        help="Directory with V-gene FASTA files (default: data/raw/positive)"
    )
    parser.add_argument(
        "--background",
        type=str,
        default="data/background/background_hard_negatives.fasta",
        help="Path to background FASTA file"
    )
    parser.add_argument(
        "--loci",
        nargs="+",
        default=["ighv", "igkv", "trav", "trbv"],
        help="Loci to include (default: ighv igkv trav trbv)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for train/val CSVs"
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=2.0,
        help="Background:V-gene ratio (default: 2.0)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=80,
        help="Minimum sequence length (default: 80 aa)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=140,
        help="Maximum sequence length (default: 140 aa)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    print("=" * 80)
    print("MULTICLASS DATASET PREPARATION - v2.0.0")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Background file: {args.background}")
    print(f"Loci: {', '.join(args.loci)}")
    print(f"Background ratio: {args.background_ratio}:1")
    print(f"Length range: {args.min_length}-{args.max_length} aa")
    print(f"Train/val split: {int((1-args.test_size)*100)}/{int(args.test_size*100)}")
    print("=" * 80)
    print()

    # Load V-genes
    print("📖 LOADING V-GENES")
    print("-" * 80)
    vgene_records = load_vgenes_simple(
        args.input_dir,
        args.loci,
        args.min_length,
        args.max_length
    )

    if not vgene_records:
        print("\n❌ Error: No V-genes loaded!")
        print("   Check that files exist: ighv.fasta, igkv.fasta, trav.fasta, trbv.fasta")
        return

    # Load background
    print()
    print("📖 LOADING BACKGROUND")
    print("-" * 80)

    num_background = int(len(vgene_records) * args.background_ratio)
    print(f"   Target background: {num_background:,} ({args.background_ratio}:1 ratio)")
    print()

    background_records = load_background(
        args.background,
        num_background,
        args.min_length,
        args.max_length,
        args.seed
    )

    if not background_records:
        print("\n❌ Error: No background sequences loaded!")
        print(f"   Generate with: python scripts/02_generate_background.py")
        return

    # Combine
    all_records = vgene_records + background_records
    df = pd.DataFrame(all_records)

    # Statistics
    print()
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Total sequences: {len(df):,}")
    print()
    print("Class distribution:")
    class_dist = df['class_name'].value_counts().sort_index()
    for class_name, count in class_dist.items():
        pct = count / len(df) * 100
        print(f"  {class_name:12s}: {count:6,} ({pct:5.1f}%)")

    print()
    print("Length statistics by class:")
    length_stats = df.groupby('class_name')['length'].describe()[['mean', 'min', 'max']]
    print(length_stats.to_string())

    # Train/val split (stratified by label)
    print()
    print("=" * 80)
    print("TRAIN/VAL SPLIT")
    print("=" * 80)

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df['label']
    )

    print(f"Train: {len(train_df):,} sequences ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df):,} sequences ({len(val_df)/len(df)*100:.1f}%)")

    print()
    print("Train class distribution:")
    train_dist = train_df['class_name'].value_counts().sort_index()
    for class_name, count in train_dist.items():
        print(f"  {class_name:12s}: {count:6,}")

    print()
    print("Val class distribution:")
    val_dist = val_df['class_name'].value_counts().sort_index()
    for class_name, count in val_dist.items():
        print(f"  {class_name:12s}: {count:6,}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    train_file = os.path.join(args.output_dir, "train_multispecies_multiclass.csv")
    val_file = os.path.join(args.output_dir, "val_multispecies_multiclass.csv")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print()
    print("=" * 80)
    print("✅ DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"💾 Train: {train_file}")
    print(f"💾 Val:   {val_file}")
    print()
    print("Next step: Train multiclass model with terminal-region encoding")
    print("  python scripts/03_train_model.py \\")
    print(f"      --train-csv {train_file} \\")
    print(f"      --val-csv {val_file}")
    print()


if __name__ == "__main__":
    main()
