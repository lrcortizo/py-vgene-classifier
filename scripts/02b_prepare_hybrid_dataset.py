#!/usr/bin/env python3
"""
Create hybrid dataset: V-genes + Hard Negatives + Synthetic Background
Combines the best of all approaches for robust training.
"""

import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from Bio import SeqIO

def main():
    parser = argparse.ArgumentParser(
        description="Create hybrid dataset with hard negatives"
    )
    parser.add_argument(
        "--vgenes-csv",
        type=Path,
        default="data/processed/train_multispecies_multiclass.csv",
        help="Original dataset with V-genes and synthetic background"
    )
    parser.add_argument(
        "--hard-negatives",
        type=Path,
        default="data/background_extended/background_hard_negatives_8k.fasta",
        help="Hard negative sequences (FASTA)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/processed_hybrid",
        help="Output directory"
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=22000,
        help="Number of synthetic sequences to keep (default: 22000)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CREATE HYBRID DATASET - v2.0.0")
    print("=" * 80)
    print(f"Strategy: V-genes + Hard Negatives + Synthetic")
    print("=" * 80)
    print()

    # Load original dataset
    print("📖 Loading original dataset...")
    df_orig = pd.read_csv(args.vgenes_csv)
    print(f"   Total: {len(df_orig):,} sequences")

    # Separate V-genes and background
    vgenes = df_orig[df_orig['class_name'] != 'background'].copy()
    background_synthetic = df_orig[df_orig['class_name'] == 'background'].copy()

    print(f"\n   V-genes: {len(vgenes):,}")
    print(f"   Synthetic background: {len(background_synthetic):,}")

    # Sample synthetic background
    if len(background_synthetic) > args.n_synthetic:
        background_synthetic = background_synthetic.sample(
            n=args.n_synthetic,
            random_state=args.seed
        )
        print(f"   Sampled {args.n_synthetic:,} synthetic sequences")

    # Load hard negatives
    print(f"\n📖 Loading hard negatives...")
    print(f"   From: {args.hard_negatives}")

    hard_neg_records = []
    for rec in SeqIO.parse(args.hard_negatives, 'fasta'):
        hard_neg_records.append({
            'id': rec.id,
            'sequence': str(rec.seq),
            'length': len(rec.seq),
            'species': 'hard_negative',
            'locus': 'background',
            'label': 0,
            'class_name': 'background'
        })

    hard_negatives = pd.DataFrame(hard_neg_records)
    print(f"   Hard negatives: {len(hard_negatives):,}")

    # Combine all
    print(f"\n🔗 Combining datasets...")
    df_combined = pd.concat([
        vgenes,
        hard_negatives,
        background_synthetic
    ], ignore_index=True)

    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Statistics
    print(f"\n📊 COMBINED DATASET STATISTICS")
    print("=" * 80)
    print(f"Total sequences: {len(df_combined):,}")
    print()
    print("Class distribution:")
    class_dist = df_combined['class_name'].value_counts()
    for class_name, count in class_dist.items():
        pct = count / len(df_combined) * 100
        print(f"  {class_name:12s}: {count:6,} ({pct:5.1f}%)")

    # Background breakdown
    bg_total = class_dist.get('background', 0)
    if bg_total > 0:
        print(f"\nBackground breakdown:")
        print(f"  Hard negatives: {len(hard_negatives):6,} ({len(hard_negatives)/bg_total*100:5.1f}% of background)")
        print(f"  Synthetic:      {len(background_synthetic):6,} ({len(background_synthetic)/bg_total*100:5.1f}% of background)")

    # Train/val split
    print(f"\n✂️  Creating train/val split ({int((1-args.test_size)*100)}/{int(args.test_size*100)})...")

    train_df, val_df = train_test_split(
        df_combined,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df_combined['label']
    )

    print(f"\n📊 SPLIT STATISTICS")
    print("=" * 80)
    print(f"Train: {len(train_df):,} sequences")
    train_dist = train_df['class_name'].value_counts()
    for class_name, count in train_dist.items():
        print(f"  {class_name:12s}: {count:6,}")

    print(f"\nVal:   {len(val_df):,} sequences")
    val_dist = val_df['class_name'].value_counts()
    for class_name, count in val_dist.items():
        print(f"  {class_name:12s}: {count:6,}")

    # Save
    train_file = args.output_dir / "train_hybrid.csv"
    val_file = args.output_dir / "val_hybrid.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print()
    print("=" * 80)
    print("✅ HYBRID DATASET CREATED")
    print("=" * 80)
    print(f"💾 Train: {train_file}")
    print(f"💾 Val:   {val_file}")
    print()
    print("Next step: Train model")
    print(f"  python scripts/03_train_model.py \\")
    print(f"      --train-csv {train_file} \\")
    print(f"      --val-csv {val_file} \\")
    print(f"      --output-dir models/v2_hybrid \\")
    print(f"      --epochs 30")
    print()


if __name__ == "__main__":
    main()
