"""
Script 01: Generate Hard Negative Background Sequences
Version: 2.0.0
Purpose: Generate high-quality negative examples from Ig-superfamily proteins

BACKGROUND STRATEGY:
Instead of random genomic translations, this script extracts sequences that are
structurally similar to V-genes but functionally different. This forces the
classifier to learn discriminative features rather than just random vs V-gene.

SOURCES OF HARD NEGATIVES:
1. MHC Class I and II proteins (similar Ig-fold structure)
2. Constant regions (C-genes from IG loci)
3. Joining segments (J-genes from IG/TCR loci)
4. Other Ig-superfamily proteins (CD28, CTLA4, NITR, etc.)

USAGE:
    python scripts/01_generate_background.py --sources mhc,c_regions,ig_superfamily
"""

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from pathlib import Path
import argparse
import time
import os
from dotenv import load_dotenv
from collections import Counter

# Load environment
load_dotenv()
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
if not NCBI_EMAIL:
    raise ValueError("NCBI_EMAIL not found. Create .env file with your email.")
Entrez.email = NCBI_EMAIL

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "background"

# Hard negative sources
HARD_NEGATIVE_SOURCES = {
    "mhc": {
        "name": "MHC Class I/II",
        "queries": [
            "HLA-A[Gene Name] AND human[Organism]",
            "HLA-B[Gene Name] AND human[Organism]",
            "HLA-C[Gene Name] AND human[Organism]",
            "HLA-DRA[Gene Name] AND human[Organism]",
            "HLA-DRB[Gene Name] AND human[Organism]",
        ],
        "max_per_query": 20,
        "description": "MHC proteins have Ig-like domains similar to V-genes",
    },
    "c_regions": {
        "name": "Constant Regions",
        "queries": [
            "IGHG[Gene Name] AND human[Organism]",
            "IGHA[Gene Name] AND human[Organism]",
            "IGHM[Gene Name] AND human[Organism]",
            "IGKC[Gene Name] AND human[Organism]",
            "IGLC[Gene Name] AND human[Organism]",
        ],
        "max_per_query": 15,
        "description": "C-regions from IG loci (structurally related to V-genes)",
    },
    "ig_superfamily": {
        "name": "Ig Superfamily",
        "queries": [
            "CD28[Gene Name] AND human[Organism]",
            "CTLA4[Gene Name] AND human[Organism]",
            "ICOS[Gene Name] AND human[Organism]",
            "PD1[Gene Name] AND human[Organism]",
            "PDL1[Gene Name] AND human[Organism]",
        ],
        "max_per_query": 10,
        "description": "Other Ig-superfamily proteins with similar domains",
    },
}

# Sequence quality filters
MIN_LENGTH = 80   # Minimum for terminal-region encoding
MAX_LENGTH = 130  # Slightly above max V-gene length
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")


def fetch_protein_sequences(query, max_results=20):
    """
    Fetch protein sequences from NCBI Protein database.

    Args:
        query: NCBI search query
        max_results: Maximum sequences to retrieve

    Returns:
        List of SeqRecord objects
    """
    try:
        # Search
        handle = Entrez.esearch(db="protein", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]

        if not id_list:
            return []

        # Fetch sequences
        handle = Entrez.efetch(db="protein", id=id_list, rettype="fasta", retmode="text")
        sequences = list(SeqIO.parse(handle, "fasta"))
        handle.close()

        return sequences

    except Exception as e:
        print(f"   ❌ Error fetching '{query}': {str(e)[:60]}")
        return []


def filter_sequence(seq, min_len=MIN_LENGTH, max_len=MAX_LENGTH):
    """
    Filter sequence by quality criteria.

    Args:
        seq: SeqRecord object
        min_len: Minimum length
        max_len: Maximum length

    Returns:
        SeqRecord or None if filtered out
    """
    seq_str = str(seq.seq).upper()

    # Length check
    if not (min_len <= len(seq_str) <= max_len):
        return None

    # Check for invalid characters
    if not all(aa in ALLOWED_AA for aa in seq_str):
        return None

    # Check for excessive repeats (low complexity)
    # If any single AA is >40%, likely low complexity
    aa_counts = Counter(seq_str)
    max_freq = max(aa_counts.values()) / len(seq_str)
    if max_freq > 0.4:
        return None

    return seq


def extract_domains(seq, target_length_range=(80, 130)):
    """
    Extract Ig-like domains from a protein sequence.

    For long proteins (e.g., full MHC), extract individual domains
    similar in length to V-genes.

    Args:
        seq: SeqRecord with full protein
        target_length_range: Tuple of (min, max) domain length

    Returns:
        List of SeqRecord objects (domains)
    """
    seq_str = str(seq.seq)
    min_len, max_len = target_length_range

    # If sequence is already in target range, return as-is
    if min_len <= len(seq_str) <= max_len:
        return [seq]

    # If too short, skip
    if len(seq_str) < min_len:
        return []

    # If too long, extract sliding windows
    domains = []
    step = 20  # Overlap windows

    for i in range(0, len(seq_str) - min_len + 1, step):
        for length in range(min_len, min(max_len + 1, len(seq_str) - i + 1)):
            domain_seq = seq_str[i:i+length]

            # Create new record
            domain_record = SeqIO.SeqRecord(
                Seq(domain_seq),
                id=f"{seq.id}_dom_{i}_{length}",
                description=f"Domain from {seq.description}"
            )

            # Filter
            if filter_sequence(domain_record):
                domains.append(domain_record)
                break  # Take first valid length for this position

        # Limit domains per protein
        if len(domains) >= 3:
            break

    return domains


def generate_hard_negatives(sources, target_total=400):
    """
    Generate hard negative sequences from specified sources.

    Args:
        sources: List of source keys (e.g., ['mhc', 'c_regions'])
        target_total: Target total number of sequences

    Returns:
        Dict of source -> list of sequences
    """
    all_sequences = {}
    target_per_source = target_total // len(sources)

    for source_key in sources:
        if source_key not in HARD_NEGATIVE_SOURCES:
            print(f"⚠️  Unknown source: {source_key}")
            continue

        source_config = HARD_NEGATIVE_SOURCES[source_key]
        print()
        print(f"{'='*80}")
        print(f"SOURCE: {source_config['name']}")
        print(f"{'='*80}")
        print(f"Description: {source_config['description']}")
        print(f"Target sequences: {target_per_source}")
        print()

        source_sequences = []

        for query in source_config['queries']:
            print(f"  Query: {query[:60]}...", end="")

            # Fetch
            raw_sequences = fetch_protein_sequences(
                query,
                max_results=source_config['max_per_query']
            )

            if not raw_sequences:
                print(" ⚠️  No results")
                continue

            print(f" ✅ {len(raw_sequences)} proteins")

            # Extract domains and filter
            for seq in raw_sequences:
                domains = extract_domains(seq)
                source_sequences.extend(domains)

            # Rate limiting
            time.sleep(0.4)

            # Check if we have enough
            if len(source_sequences) >= target_per_source:
                break

        # Trim to target
        source_sequences = source_sequences[:target_per_source]
        all_sequences[source_key] = source_sequences

        print()
        print(f"  ✅ Collected {len(source_sequences)} sequences from {source_config['name']}")

    return all_sequences


def save_sequences(sequences_by_source, output_dir):
    """
    Save sequences to files.

    Args:
        sequences_by_source: Dict of source -> sequences
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual source files
    for source_key, sequences in sequences_by_source.items():
        if sequences:
            output_file = output_dir / f"background_{source_key}.fasta"
            SeqIO.write(sequences, output_file, "fasta")
            print(f"  💾 {source_key}: {len(sequences)} sequences → {output_file.name}")

    # Save combined file
    all_sequences = []
    for sequences in sequences_by_source.values():
        all_sequences.extend(sequences)

    if all_sequences:
        combined_file = output_dir / "background_hard_negatives.fasta"
        SeqIO.write(all_sequences, combined_file, "fasta")
        print()
        print(f"  💾 Combined: {len(all_sequences)} sequences → {combined_file.name}")

        # Quality summary
        lengths = [len(seq.seq) for seq in all_sequences]
        print()
        print(f"  📊 Quality Summary:")
        print(f"     Length range: {min(lengths)}-{max(lengths)} aa")
        print(f"     Mean length: {sum(lengths)/len(lengths):.1f} aa")
        print(f"     Terminal-encoding ready (≥80 aa): {len(all_sequences)} (100%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard negative background sequences for V-gene classification"
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="mhc,c_regions,ig_superfamily",
        help="Comma-separated list of sources: mhc, c_regions, ig_superfamily"
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=400,
        help="Target total number of background sequences (default: 400 for 2:1 ratio)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for background files"
    )

    args = parser.parse_args()

    # Parse sources
    sources = [s.strip() for s in args.sources.split(",")]

    print("=" * 80)
    print("HARD NEGATIVE BACKGROUND GENERATION - v2.0.0")
    print("=" * 80)
    print()
    print(f"Sources: {', '.join(sources)}")
    print(f"Target total: {args.target_total} sequences")
    print(f"Output directory: {args.output_dir}")
    print()
    print("RATIONALE:")
    print("  Hard negatives are structurally similar to V-genes but functionally")
    print("  different. This forces the classifier to learn discriminative features")
    print("  instead of just distinguishing random sequences from V-genes.")
    print("=" * 80)

    # Generate
    sequences_by_source = generate_hard_negatives(sources, args.target_total)

    # Save
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    save_sequences(sequences_by_source, args.output_dir)

    print()
    print("=" * 80)
    print("✅ HARD NEGATIVE GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review generated sequences in data/background/")
    print("  2. Use in training: scripts/02_prepare_dataset.py")
    print("  3. Specify background file with --background flag")
    print()


if __name__ == "__main__":
    main()
