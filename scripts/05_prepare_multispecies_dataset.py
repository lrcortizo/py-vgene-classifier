#!/usr/bin/env python3
"""
Prepare multi-species V-gene training dataset.

This script:
1. Reads annotated V-gene FASTAs from multiple species
2. Extracts clean sequences (removes gaps/annotations)
3. Uses synthetic or existing background sequences
4. Creates train/validation split

Usage:
    python scripts/05_prepare_multispecies_dataset.py \
        --input-dir path/to/annotated_fastas \
        --loci ighv igkv trav trbv \
        --output-dir data/processed
"""

import argparse
import random
from pathlib import Path
from typing import List, Dict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_annotated_header(header: str) -> Dict[str, str]:
    """
    Parse annotated FASTA header.
    Format: >ID|Species-Accession|locus FR1:X-Y CDR1:X-Y ...

    Returns dict with: id, species, accession, locus
    """
    parts = header.split()

    # First part: >ID|Species-Accession|locus
    main_parts = parts[0].split("|")

    info = {
        "id": main_parts[0].replace(">", ""),
        "species": "unknown",
        "accession": "unknown",
        "locus": "unknown"
    }

    if len(main_parts) >= 2:
        species_acc = main_parts[1].split("-")
        info["species"] = species_acc[0] if species_acc else "unknown"
        if len(species_acc) > 1:
            info["accession"] = "-".join(species_acc[1:])

    if len(main_parts) >= 3:
        info["locus"] = main_parts[2]

    return info


def clean_sequence(seq_str: str) -> str:
    """Remove gaps and clean sequence."""
    # Remove alignment gaps
    cleaned = seq_str.replace("-", "").replace(".", "")
    # Remove any remaining non-standard amino acids
    cleaned = "".join(c for c in cleaned if c.isalpha())
    return cleaned


def load_vgenes(input_dir: Path, loci: List[str]) -> List[SeqRecord]:
    """Load V-genes from annotated FASTA files."""
    records = []

    for locus in loci:
        # Look for files matching pattern (exclude .raw)
        pattern = f"*{locus}*.realigned.annotated.fasta"
        files = list(input_dir.glob(f"**/{pattern}"))

        # Exclude .raw files
        files = [f for f in files if ".raw." not in str(f)]

        if not files:
            print(f"⚠️  No files found for {locus}")
            continue

        for fasta_file in files:
            print(f"📖 Reading {fasta_file.name}...")
            count = 0

            for record in SeqIO.parse(fasta_file, "fasta"):
                # Parse header
                info = parse_annotated_header(record.description)

                # Clean sequence
                clean_seq = clean_sequence(str(record.seq))

                # Skip if too short or too long
                if len(clean_seq) < 80 or len(clean_seq) > 140:
                    continue

                # Create new record with clean sequence
                new_record = SeqRecord(
                    Seq(clean_seq),
                    id=f"{info['id']}_{info['species']}_{info['locus']}",
                    description=f"species={info['species']} locus={info['locus']}"
                )

                records.append(new_record)
                count += 1

            print(f"   ✅ Loaded {count} sequences")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-species V-gene training dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing annotated FASTA files"
    )
    parser.add_argument(
        "--loci",
        nargs="+",
        default=["ighv", "igkv", "trav", "trbv"],
        help="Loci to include (default: ighv igkv trav trbv)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=2.0,
        help="Ratio of background to V-genes (default: 2.0)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation set size (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 70)
    print("PREPARE MULTI-SPECIES V-GENE DATASET")
    print("=" * 70)

    # Load V-genes
    print(f"\n📚 Loading V-genes from {args.input_dir}")
    print(f"   Loci: {', '.join(args.loci)}")

    vgene_records = load_vgenes(args.input_dir, args.loci)

    if not vgene_records:
        print("\n❌ No V-gene sequences loaded!")
        return

    print(f"\n✅ Total V-genes loaded: {len(vgene_records)}")

    # Length statistics
    lengths = [len(rec.seq) for rec in vgene_records]
    print(f"   Length range: {min(lengths)}-{max(lengths)} aa")
    print(f"   Mean length: {sum(lengths)/len(lengths):.1f} aa")

    # Load background
    num_background = int(len(vgene_records) * args.background_ratio)

    print(f"\n📥 Loading background sequences...")
    print(f"   Need: {num_background} sequences (ratio {args.background_ratio}:1)")

    # Try synthetic first (as recommended by supervisor for initial phase)
    bg_file = Path("data/raw/negative/background_synthetic.fasta")
    if not bg_file.exists():
        bg_file = Path("data/raw/negative/background_uniprot.fasta")
    if not bg_file.exists():
        bg_file = Path("data/raw/negative/background.fasta")

    if bg_file.exists():
        print(f"   Using: {bg_file.name}")
        background_records = list(SeqIO.parse(bg_file, "fasta"))
        print(f"   ✅ Loaded {len(background_records)} from {bg_file}")

        # Sample if we have more than needed
        if len(background_records) > num_background:
            background_records = random.sample(background_records, num_background)
            print(f"   📊 Sampled {num_background} sequences")
        elif len(background_records) < num_background:
            print(f"   ⚠️  Only {len(background_records)} available (need {num_background})")
            print(f"   Using all available sequences")
    else:
        print(f"\n❌ No background file found!")
        print(f"\nPlease generate background first:")
        print(f"  python scripts/05b_generate_synthetic_background.py")
        return

    # Create DataFrame
    print(f"\n📊 Creating dataset...")
    data = []

    for rec in vgene_records:
        data.append({
            "id": rec.id,
            "sequence": str(rec.seq),
            "length": len(rec.seq),
            "label": 1
        })

    for rec in background_records:
        data.append({
            "id": rec.id,
            "sequence": str(rec.seq),
            "length": len(rec.seq),
            "label": 0
        })

    df = pd.DataFrame(data)

    print(f"   V-genes: {(df['label']==1).sum()}")
    print(f"   Background: {(df['label']==0).sum()}")
    print(f"   Total: {len(df)}")
    print(f"   Ratio: {(df['label']==0).sum() / (df['label']==1).sum():.2f}:1")

    # Train/val split
    print(f"\n✂️  Splitting dataset (test_size={args.test_size})...")
    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"]
    )

    print(f"   Train: {len(train_df)} ({(train_df['label']==1).sum()} V-genes)")
    print(f"   Val: {len(val_df)} ({(val_df['label']==1).sum()} V-genes)")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.output_dir / "train_multispecies.csv"
    val_csv = args.output_dir / "val_multispecies.csv"
    full_csv = args.output_dir / "full_dataset_multispecies.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    df.to_csv(full_csv, index=False)

    print(f"\n💾 Saved:")
    print(f"   {train_csv}")
    print(f"   {val_csv}")
    print(f"   {full_csv}")

    # Also save as FASTA
    train_fasta = args.output_dir / "train_multispecies.fasta"
    val_fasta = args.output_dir / "val_multispecies.fasta"

    train_records = [
        SeqRecord(Seq(row["sequence"]), id=row["id"], description="")
        for _, row in train_df.iterrows()
    ]
    val_records = [
        SeqRecord(Seq(row["sequence"]), id=row["id"], description="")
        for _, row in val_df.iterrows()
    ]

    SeqIO.write(train_records, train_fasta, "fasta")
    SeqIO.write(val_records, val_fasta, "fasta")

    print(f"   {train_fasta}")
    print(f"   {val_fasta}")

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print("\nNext step:")
    print("  python scripts/06_train_multispecies_cnn.py")


if __name__ == "__main__":
    main()
