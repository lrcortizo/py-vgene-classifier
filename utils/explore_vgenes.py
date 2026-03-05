"""
Script 01: Explore V-gene FASTA files
Version: 2.0.0
Purpose: Initial data exploration and quality checks for V-gene classification

This script analyzes all available V-gene loci and provides statistics
for training data preparation. Currently, the multiclass pipeline uses
4 main loci (IGHV, IGKV, TRAV, TRBV) but this script supports all 7.
"""

from Bio import SeqIO
from pathlib import Path
import pandas as pd
from collections import Counter

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "positive"

# All available V-gene loci
# Primary loci used in v2.0.0 multiclass pipeline: IGHV, IGKV, TRAV, TRBV
# Additional loci available for future work: IGLV, TRDV, TRGV
ALL_LOCI = {
    "ighv": {"name": "IGHV", "type": "IG Heavy", "primary": True},
    "igkv": {"name": "IGKV", "type": "IG Kappa Light", "primary": True},
    "iglv": {"name": "IGLV", "type": "IG Lambda Light", "primary": False},
    "trav": {"name": "TRAV", "type": "TCR Alpha", "primary": True},
    "trbv": {"name": "TRBV", "type": "TCR Beta", "primary": True},
    "trdv": {"name": "TRDV", "type": "TCR Delta", "primary": False},
    "trgv": {"name": "TRGV", "type": "TCR Gamma", "primary": False},
}

# Conserved motifs commonly found in V-gene C-terminal regions
# Based on known functional V-gene signatures (typically in last ~35 aa)
CONSERVED_MOTIFS = ['YYC', 'YFC', 'YLC', 'YIC', 'YHC', 'TFC']


def analyze_fasta(fasta_path, locus_info):
    """
    Analyze a FASTA file and return comprehensive statistics.

    Args:
        fasta_path: Path to FASTA file
        locus_info: Dict with locus metadata

    Returns:
        Tuple of (stats dict, motif presence dict)
    """
    sequences = list(SeqIO.parse(fasta_path, "fasta"))

    if not sequences:
        return None, None

    lengths = [len(seq.seq) for seq in sequences]

    # Check suitability for terminal-region encoding methods
    # (requires sufficient length for N-terminal and C-terminal features)
    min_length_terminal_encoding = 80
    suitable_terminal = sum(1 for seq in sequences
                           if len(seq.seq) >= min_length_terminal_encoding)

    # Check for conserved motifs (typically in C-terminal region)
    motif_presence = {}
    c_terminal_window = 35  # Standard window for C-terminal analysis
    for motif in CONSERVED_MOTIFS:
        # Check C-terminal region for conserved signatures
        count = sum(1 for seq in sequences
                   if len(seq.seq) >= c_terminal_window
                   and motif in str(seq.seq[-c_terminal_window:]))
        motif_presence[motif] = count

    stats = {
        "locus": locus_info["name"],
        "type": locus_info["type"],
        "primary": "✓" if locus_info["primary"] else "",
        "num_sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": round(sum(lengths) / len(lengths), 1),
        "median_length": sorted(lengths)[len(lengths) // 2],
        "suitable_terminal_encoding": suitable_terminal,
        "terminal_encoding_pct": round(suitable_terminal / len(sequences) * 100, 1),
        "with_motifs": sum(motif_presence.values()),
        "motif_coverage": round(
            sum(1 for seq in sequences
                if len(seq.seq) >= c_terminal_window
                and any(m in str(seq.seq[-c_terminal_window:])
                       for m in CONSERVED_MOTIFS))
            / len(sequences) * 100, 1
        ) if len(sequences) > 0 else 0,
    }

    return stats, motif_presence


def main():
    print("=" * 80)
    print("V-GENE EXPLORATION - v2.0.0")
    print("=" * 80)
    print("Analysis of V-gene loci for multiclass classification pipeline")
    print()
    print("Primary loci (used in v2.0.0): IGHV, IGKV, TRAV, TRBV")
    print("Additional loci (available): IGLV, TRDV, TRGV")
    print("=" * 80)
    print()

    all_stats = []
    primary_sequences = 0
    total_sequences = 0

    # Track which loci were found
    found_loci = []
    missing_loci = []

    for locus_key, locus_info in ALL_LOCI.items():
        fasta_file = f"{locus_key}.fasta"
        fasta_path = DATA_DIR / fasta_file

        if not fasta_path.exists():
            print(f"⚠️  {locus_info['name']:8s} ({locus_info['type']:20s}) - NOT FOUND")
            missing_loci.append(locus_info['name'])
            continue

        stats, motifs = analyze_fasta(fasta_path, locus_info)

        if stats:
            all_stats.append(stats)
            found_loci.append(locus_info['name'])
            total_sequences += stats["num_sequences"]

            if locus_info["primary"]:
                primary_sequences += stats["num_sequences"]

            # Print locus info
            primary_marker = "⭐" if locus_info["primary"] else "  "
            print(f"{primary_marker} 📁 {stats['locus']:<8s} ({stats['type']:20s})")
            print(f"      Sequences: {stats['num_sequences']:>6,}")
            print(f"      Length: {stats['min_length']:>3}-{stats['max_length']:<3} aa "
                  f"(mean: {stats['mean_length']:>5.1f} aa)")
            print(f"      Terminal-encoding ready: {stats['suitable_terminal_encoding']:>5}/{stats['num_sequences']:<5} "
                  f"({stats['terminal_encoding_pct']:>5.1f}%)")
            print(f"      Conserved motifs: {stats['with_motifs']:>4} occurrences "
                  f"({stats['motif_coverage']:>5.1f}% sequences)")

            # Show most common motif
            if motifs:
                top_motif = max(motifs.items(), key=lambda x: x[1])
                if top_motif[1] > 0:
                    print(f"      Most common: {top_motif[0]} "
                          f"({top_motif[1]} sequences, "
                          f"{top_motif[1]/stats['num_sequences']*100:.1f}%)")
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Loci found: {len(found_loci)}/{len(ALL_LOCI)}")
    print(f"  Found: {', '.join(found_loci)}")
    if missing_loci:
        print(f"  Missing: {', '.join(missing_loci)}")
    print()
    print(f"Total sequences: {total_sequences:,}")
    print(f"  Primary loci (IGHV, IGKV, TRAV, TRBV): {primary_sequences:,}")
    print(f"  Additional loci: {total_sequences - primary_sequences:,}")
    print()
    print(f"Background sequences recommended:")
    print(f"  For primary loci (2:1 ratio): {primary_sequences * 2:,}")
    print(f"  For all loci (2:1 ratio): {total_sequences * 2:,}")
    print()

    # Save statistics
    if all_stats:
        df = pd.DataFrame(all_stats)

        # Reorder columns for better readability
        column_order = [
            "locus", "type", "primary", "num_sequences",
            "min_length", "max_length", "mean_length", "median_length",
            "suitable_terminal_encoding", "terminal_encoding_pct",
            "with_motifs", "motif_coverage"
        ]
        df = df[column_order]

        print("=" * 80)
        print("STATISTICS TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        print()

        # Save to CSV
        output_path = Path(__file__).parent.parent / "results" / "vgene_stats.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"💾 Statistics saved to: {output_path}")
        print()

        # Quality checks
        print("=" * 80)
        print("QUALITY CHECKS")
        print("=" * 80)

        # Terminal-region encoding suitability
        primary_df = df[df['primary'] == '✓']
        if len(primary_df) > 0:
            avg_terminal = primary_df['terminal_encoding_pct'].mean()
            if avg_terminal >= 95:
                print(f"✅ Terminal-region encoding (primary loci): {avg_terminal:.1f}% sequences suitable")
            elif avg_terminal >= 90:
                print(f"⚠️  Terminal-region encoding (primary loci): {avg_terminal:.1f}% sequences suitable")
                print(f"    Some sequences <80 aa may have reduced encoding quality")
            else:
                print(f"❌ Terminal-region encoding (primary loci): {avg_terminal:.1f}% sequences suitable")
                print(f"    Many sequences too short - consider filtering or using alternative encoding")

        # Conserved motifs
        if len(primary_df) > 0:
            avg_motif = primary_df['motif_coverage'].mean()
            if avg_motif >= 80:
                print(f"✅ Conserved motifs (primary loci): {avg_motif:.1f}% coverage")
            elif avg_motif >= 60:
                print(f"⚠️  Conserved motifs (primary loci): {avg_motif:.1f}% coverage")
                print(f"    Lower than expected - some genes may lack canonical C-terminal motifs")
            else:
                print(f"❌ Conserved motifs (primary loci): {avg_motif:.1f}% coverage")
                print(f"    Very low coverage - verify sequence quality and orientation")

        # Length distribution
        if len(primary_df) > 0:
            avg_length = primary_df['mean_length'].mean()
            print(f"\n📏 Average V-gene length (primary loci): {avg_length:.1f} aa")
            if 100 <= avg_length <= 120:
                print(f"   ✅ Within typical V-gene range (100-120 aa)")
            else:
                print(f"   ⚠️  Outside typical range - verify sequences")

        print()
        print("=" * 80)
        print("NOTES")
        print("=" * 80)
        print("• Primary loci (⭐) are used in the current v2.0.0 multiclass pipeline")
        print("• Additional loci can be included by updating training scripts")
        print("• Terminal-region encoding requires ≥80 aa for reliable N/C-terminal features")
        print("• Conserved motifs (YYC, YFC, etc.) are expected in functional V-genes")
        print("=" * 80)


if __name__ == "__main__":
    main()
