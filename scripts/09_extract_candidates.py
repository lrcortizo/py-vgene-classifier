#!/usr/bin/env python3
"""
Extract candidate V-gene sequences from TBLASTN results.

This script:
1. Parses TBLASTN output
2. Extends hits to capture full V-gene region
3. Translates DNA to protein
4. Removes duplicates
5. Saves candidate sequences

Usage:
    python scripts/09_extract_candidates.py \
        --tblastn-results results/bat/tblastn_results.txt \
        --genome data/genomes/myotis_lucifugus/GCF_000147115.1.fna \
        --output results/bat/candidates.fasta \
        --min-length 80 \
        --extend 150
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict


def parse_tblastn_results(
    results_file: Path,
    min_length: int = 80
) -> List[Tuple[str, int, int, str, float]]:
    """
    Parse TBLASTN results.

    Returns list of (contig_id, start, end, strand, evalue)
    """
    hits = []

    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue

            fields = line.strip().split('\t')

            # Format: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen
            query_id = fields[0]
            contig_id = fields[1]
            pident = float(fields[2])
            length = int(fields[3])
            sstart = int(fields[8])
            send = int(fields[9])
            evalue = float(fields[10])

            # Skip short hits
            if length < min_length:
                continue

            # Determine strand and coordinates
            if sstart < send:
                start = sstart
                end = send
                strand = '+'
            else:
                start = send
                end = sstart
                strand = '-'

            hits.append((contig_id, start, end, strand, evalue))

    return hits


def merge_overlapping_hits(
    hits: List[Tuple[str, int, int, str, float]]
) -> List[Tuple[str, int, int, str]]:
    """
    Merge overlapping hits on the same contig and strand.

    Returns merged list of (contig_id, start, end, strand)
    """
    # Group by contig and strand
    grouped = defaultdict(list)
    for contig_id, start, end, strand, evalue in hits:
        grouped[(contig_id, strand)].append((start, end, evalue))

    merged = []

    for (contig_id, strand), coords in grouped.items():
        # Sort by start position
        coords.sort(key=lambda x: x[0])

        # Merge overlapping regions
        current_start, current_end, min_evalue = coords[0]

        for start, end, evalue in coords[1:]:
            if start <= current_end + 50:  # Allow 50 bp gap
                current_end = max(current_end, end)
                min_evalue = min(min_evalue, evalue)
            else:
                merged.append((contig_id, current_start, current_end, strand))
                current_start, current_end, min_evalue = start, end, evalue

        merged.append((contig_id, current_start, current_end, strand))

    return merged


def extract_sequences(
    genome_file: Path,
    regions: List[Tuple[str, int, int, str]],
    extend: int = 150
) -> List[SeqRecord]:
    """
    Extract and translate sequences from genome.

    Args:
        genome_file: Genome FASTA file
        regions: List of (contig_id, start, end, strand)
        extend: Bases to extend on each side

    Returns list of protein SeqRecord objects
    """
    # Load genome
    print(f"\n📖 Loading genome...")
    genome_raw = {}
    for record in SeqIO.parse(genome_file, "fasta"):
        # Store with first word of ID (e.g., "NW_005871048.1")
        clean_id = record.id.split()[0]
        genome_raw[clean_id] = record

    print(f"   ✅ Loaded {len(genome_raw)} contigs")

    # Create lookup with multiple ID formats
    print(f"   🔍 Creating ID lookup...")
    genome = {}
    for contig_id, record in genome_raw.items():
        genome[contig_id] = record.seq

        # Also store with BLAST format: ref|ID|
        genome[f"ref|{contig_id}|"] = record.seq
        genome[f"gb|{contig_id}|"] = record.seq
        genome[f"emb|{contig_id}|"] = record.seq

        # Without version number
        if '.' in contig_id:
            base_id = contig_id.split('.')[0]
            genome[base_id] = record.seq
            genome[f"ref|{base_id}|"] = record.seq

    print(f"   ✅ ID lookup ready ({len(genome)} entries)")

    # Extract sequences
    print(f"\n🧬 Extracting {len(regions)} candidate regions...")

    candidates = []
    failed = 0

    for idx, (contig_id, start, end, strand) in enumerate(regions):
        if contig_id not in genome:
            failed += 1
            if failed == 1:  # Debug first failure
                print(f"   ⚠️  First failed lookup: '{contig_id}'")
                print(f"   Sample available IDs: {list(genome_raw.keys())[:3]}")
            continue

        contig_seq = genome[contig_id]

        # Extend region (1-based to 0-based)
        extended_start = max(0, start - 1 - extend)
        extended_end = min(len(contig_seq), end + extend)

        # Extract DNA
        dna_seq = contig_seq[extended_start:extended_end]

        # Reverse complement if needed
        if strand == '-':
            dna_seq = dna_seq.reverse_complement()

        # Translate in all 3 frames
        best_protein = None
        best_length = 0

        for frame in range(3):
            try:
                protein_seq = dna_seq[frame:].translate(to_stop=False)
                protein_str = str(protein_seq).rstrip('*')

                # Split by stops, take longest fragment
                fragments = protein_str.split('*')

                for fragment in fragments:
                    if len(fragment) >= 70 and len(fragment) > best_length:
                        best_length = len(fragment)
                        best_protein = fragment
            except Exception:
                continue

        if best_protein:
            record = SeqRecord(
                Seq(best_protein),
                id=f"candidate_{idx+1}",
                description=f"{contig_id}:{extended_start}-{extended_end}({strand}) len={len(best_protein)}"
            )
            candidates.append(record)
        else:
            failed += 1

        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx+1}/{len(regions)}... (extracted: {len(candidates)}, failed: {failed})")

    print(f"   ✅ Extracted {len(candidates)} valid candidates (failed: {failed})")

    return candidates


def remove_duplicates(sequences: List[SeqRecord]) -> List[SeqRecord]:
    """Remove duplicate sequences (by sequence identity)."""
    print(f"\n🔍 Removing duplicates...")

    seen = set()
    unique = []

    for rec in sequences:
        seq_str = str(rec.seq)
        if seq_str not in seen:
            seen.add(seq_str)
            unique.append(rec)

    print(f"   Original: {len(sequences)}")
    print(f"   Unique: {len(unique)}")
    print(f"   Removed: {len(sequences) - len(unique)}")

    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Extract candidate V-gene sequences from TBLASTN results"
    )
    parser.add_argument(
        "--tblastn-results",
        type=Path,
        required=True,
        help="TBLASTN results file (tabular format)"
    )
    parser.add_argument(
        "--genome",
        type=Path,
        required=True,
        help="Genome FASTA file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output FASTA file for candidates"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=80,
        help="Minimum protein length (default: 80)"
    )
    parser.add_argument(
        "--extend",
        type=int,
        default=150,
        help="Bases to extend on each side of hit (default: 150)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EXTRACT V-GENE CANDIDATES")
    print("=" * 70)

    # Parse TBLASTN results
    print(f"\n📊 Parsing TBLASTN results...")
    print(f"   File: {args.tblastn_results}")

    hits = parse_tblastn_results(args.tblastn_results, args.min_length)
    print(f"   ✅ Found {len(hits)} hits")

    # Merge overlapping hits
    print(f"\n🔗 Merging overlapping hits...")
    merged = merge_overlapping_hits(hits)
    print(f"   ✅ Merged to {len(merged)} regions")

    # Extract sequences
    candidates = extract_sequences(args.genome, merged, args.extend)

    # Remove duplicates
    unique_candidates = remove_duplicates(candidates)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(unique_candidates, args.output, "fasta")

    # Summary
    lengths = [len(rec.seq) for rec in unique_candidates]

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total candidates: {len(unique_candidates)}")
    print(f"   Length range: {min(lengths)}-{max(lengths)} aa")
    print(f"   Mean length: {sum(lengths)/len(lengths):.1f} aa")
    print(f"\n💾 Saved: {args.output}")
    print("\nNext step:")
    print(f"  python scripts/10_filter_positives.py \\")
    print(f"      --candidates {args.output} \\")
    print(f"      --model models/best_model_multispecies.pt \\")
    print(f"      --output results/bat/vgenes_predicted.fasta")


if __name__ == "__main__":
    main()
