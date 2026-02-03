#!/usr/bin/env python3
"""
Prepare multi-species dataset for MULTICLASS CNN training.
Classes: 0=background, 1=IGHV, 2=IGKV, 3=TRAV, 4=TRBV
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
    'iglv': 2,  # IGLV grouped with IGKV
    'trav': 3,
    'trbv': 4,
    'background': 0,
    'non-v-gene': 0
}

CLASS_NAMES = {
    0: 'background',
    1: 'IGHV',
    2: 'IGKV',
    3: 'TRAV',
    4: 'TRBV'
}

def parse_annotated_header(header):
    """
    Parse header from annotated FASTA files.
    Format: >ID|Species-Accession|locus FR1:X-Y CDR1:X-Y...
    """
    try:
        parts = header.split("|")
        if len(parts) < 3:
            return None

        info = {
            "id": parts[0].replace(">", "").strip(),
            "species": parts[1].split("-")[0].strip() if "-" in parts[1] else parts[1].strip(),
            "locus": parts[2].split()[0].strip().lower()
        }
        return info
    except Exception as e:
        print(f"Warning: Could not parse header: {header[:50]}... Error: {e}")
        return None

def clean_sequence(seq_str):
    """Remove alignment gaps and non-alphabetic characters."""
    cleaned = seq_str.replace("-", "").replace(".", "")
    cleaned = "".join(c for c in cleaned if c.isalpha())
    return cleaned.upper()

def load_vgenes(input_dir, loci, min_length=80, max_length=140):
    """Load V-genes from annotated FASTA files."""
    vgene_records = []
    stats = {locus: 0 for locus in loci}

    for locus in loci:
        # Try different file patterns (RECURSIVAMENTE con **)
        patterns = [
            f"**/{locus}_*.realigned.annotated.fasta",
            f"**/{locus}_*.fasta",
            f"**/*{locus}*.fasta"
        ]

        files_found = []
        for pattern in patterns:
            files_found.extend(Path(input_dir).glob(pattern))

        if not files_found:
            print(f"Warning: No files found for locus {locus}")
            continue

        for fasta_file in files_found:
            print(f"Loading {fasta_file.name}...")

            for record in SeqIO.parse(fasta_file, "fasta"):
                # Parse header
                header_info = parse_annotated_header(record.description)
                if not header_info:
                    continue

                # Clean sequence
                clean_seq = clean_sequence(str(record.seq))

                # Filter by length
                if len(clean_seq) < min_length or len(clean_seq) > max_length:
                    continue

                # Get class label
                parsed_locus = header_info['locus']
                if parsed_locus not in LOCUS_TO_CLASS:
                    print(f"Warning: Unknown locus '{parsed_locus}', skipping")
                    continue

                class_label = LOCUS_TO_CLASS[parsed_locus]

                vgene_records.append({
                    "id": header_info["id"],
                    "sequence": clean_seq,
                    "length": len(clean_seq),
                    "species": header_info["species"],
                    "locus": parsed_locus,
                    "label": class_label,
                    "class_name": CLASS_NAMES[class_label]
                })

                stats[locus] += 1

    print(f"\nV-genes loaded:")
    for locus, count in stats.items():
        print(f"  {locus.upper()}: {count}")
    print(f"  Total: {len(vgene_records)}")

    return vgene_records

def load_background(background_file, num_needed, min_length=80, max_length=140, seed=42):
    """Load background sequences."""
    if not os.path.exists(background_file):
        print(f"Error: Background file not found: {background_file}")
        return []

    print(f"\nLoading background from {background_file}...")
    all_background = list(SeqIO.parse(background_file, "fasta"))

    # Filter by length
    filtered = [rec for rec in all_background
                if min_length <= len(rec.seq) <= max_length]

    print(f"  Total available: {len(all_background)}")
    print(f"  After length filter: {len(filtered)}")

    # Sample if needed
    random.seed(seed)
    if len(filtered) > num_needed:
        sampled = random.sample(filtered, num_needed)
    else:
        sampled = filtered
        print(f"  Warning: Only {len(filtered)} background sequences available (needed {num_needed})")

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

    print(f"  Background sequences: {len(background_records)}")

    return background_records

def main():
    parser = argparse.ArgumentParser(description="Prepare multiclass dataset")
    parser.add_argument("--input-dir", required=True, help="Directory with annotated V-gene FASTAs")
    parser.add_argument("--loci", nargs="+", default=["ighv", "igkv", "trav", "trbv"],
                        help="Loci to include")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--background-ratio", type=float, default=2.0,
                        help="Background:V-gene ratio")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--min-length", type=int, default=80, help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=140, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Load V-genes
    print("="*60)
    print("LOADING V-GENES")
    print("="*60)
    vgene_records = load_vgenes(args.input_dir, args.loci, args.min_length, args.max_length)

    if not vgene_records:
        print("Error: No V-genes loaded!")
        return

    # Load background
    print("\n" + "="*60)
    print("LOADING BACKGROUND")
    print("="*60)

    num_background = int(len(vgene_records) * args.background_ratio)

    # Try different background files
    bg_files = [
        "data/raw/negative/background_synthetic.fasta",
        "data/raw/negative/background_uniprot.fasta",
        "data/raw/negative/background.fasta"
    ]

    background_records = []
    for bg_file in bg_files:
        if os.path.exists(bg_file):
            background_records = load_background(bg_file, num_background,
                                                 args.min_length, args.max_length, args.seed)
            break

    if not background_records:
        print("Error: No background sequences loaded!")
        return

    # Combine
    all_records = vgene_records + background_records
    df = pd.DataFrame(all_records)

    # Statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total sequences: {len(df)}")
    print(f"\nClass distribution:")
    print(df['class_name'].value_counts())
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().sort_index())

    # Train/val split (stratified by label)
    print("\n" + "="*60)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*60)

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df['label']
    )

    print(f"Train: {len(train_df)} sequences")
    print(f"Val:   {len(val_df)} sequences")
    print(f"\nTrain class distribution:")
    print(train_df['class_name'].value_counts())
    print(f"\nVal class distribution:")
    print(val_df['class_name'].value_counts())

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    train_file = os.path.join(args.output_dir, "train_multispecies_multiclass.csv")
    val_file = os.path.join(args.output_dir, "val_multispecies_multiclass.csv")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print("\n" + "="*60)
    print("SAVED")
    print("="*60)
    print(f"Train: {train_file}")
    print(f"Val:   {val_file}")
    print("\nDone!")

if __name__ == "__main__":
    main()
