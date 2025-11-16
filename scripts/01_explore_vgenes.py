"""
Script to explore V gene FASTA files
"""

from Bio import SeqIO
from pathlib import Path
import pandas as pd

# Path to data
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "positive"

# FASTA files
fasta_files = [
    "ighv.fasta",
    "igkv.fasta",
    "iglv.fasta",
    "trav.fasta",
    "trbv.fasta",
    "trdv.fasta",
    "trgv.fasta",
]


def analyze_fasta(fasta_path):
    """Analyze a FASTA file and return statistics"""
    sequences = list(SeqIO.parse(fasta_path, "fasta"))

    if not sequences:
        return None

    lengths = [len(seq.seq) for seq in sequences]

    stats = {
        "file": fasta_path.name,
        "num_sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }

    return stats


# Analyze all files
print("=" * 60)
print("V GENE ANALYSIS - POSITIVE CLASS")
print("=" * 60)

all_stats = []
total_sequences = 0

for fasta_file in fasta_files:
    fasta_path = DATA_DIR / fasta_file

    if not fasta_path.exists():
        print(f"⚠️  {fasta_file} - NOT FOUND")
        continue

    stats = analyze_fasta(fasta_path)

    if stats:
        all_stats.append(stats)
        total_sequences += stats["num_sequences"]

        print(f"\n📁 {stats['file']}")
        print(f"   Sequences: {stats['num_sequences']}")
        print(f"   Length range: {stats['min_length']}-{stats['max_length']} bp")
        print(f"   Mean: {stats['mean_length']:.1f} bp")

print("\n" + "=" * 60)
print(f"TOTAL POSITIVE SEQUENCES: {total_sequences}")
print(f"NEGATIVE SEQUENCES NEEDED (3:1 ratio): {total_sequences * 3}")
print("=" * 60)

# Save statistics
if all_stats:
    df = pd.DataFrame(all_stats)
    print("\n📊 Summary:")
    print(df.to_string(index=False))

    # Save to CSV
    output_path = Path(__file__).parent.parent / "results" / "vgene_stats.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Statistics saved to: {output_path}")
