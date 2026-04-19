#!/usr/bin/env python3
"""
Script 06: Extract Candidate V-Gene Sequences
Extract candidate V-gene sequences from TBLASTN results - v2.0 with auto-cleaning

This script:
1. Parses TBLASTN output
2. Extends hits to capture full V-gene region
3. Translates DNA to protein
4. **CLEANS N-terminal** (Framework 1 detection)
5. **CLEANS C-terminal** (typical V-gene length ~95-120 aa)
6. Removes duplicates
7. Saves candidate sequences

Improvements over v1:
- Detects Framework 1 motifs to find true V-gene start
- Trims upstream junk sequences
- Limits C-terminal to typical V-gene length
- Results in cleaner sequences for classification

Usage:
    python scripts/06_extract_candidates.py \
        --tblastn-results results/bat/tblastn_results.txt \
        --genome data/genomes/myotis_lucifugus/GCF_000147115.1.fna \
        --output results/bat/candidates.fasta \
        --min-length 80 \
        --extend 150 \
        --clean-terminals
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import re


# Framework 1 motifs for V-gene start detection
# IG patterns are listed FIRST — find_vgene_start() returns on the first match,
# so IG candidates are protected from false-positive TCR matches.
#
# TCR patterns derived empirically from 3,974 TRAV and 2,626 TRBV training
# sequences (92 mammalian species). Coverage: TRAV 68.5%, TRBV 77.8%.
# False-positive rate on IG sequences: IGHV 0.3%, IGKV 0.4%.
VGENE_START_PATTERNS = [
    # ── IGHV ────────────────────────────────────────────────────────────────
    r'[QE]VQ[LV]',              # EVQL, QVQL
    # ── IGKV / IGLV ─────────────────────────────────────────────────────────
    r'D[ILV][VKQ][MLV]TQ',     # DIVMTQ, DIKMTQ, DIQMTQ
    r'Q[AS]VL[TV]Q',            # QSVLTQ, QAVLTQ
    r'Q[AS]V[LV]TQ',            # QSVVTQ, QAVVTQ
    # ── TRAV ────────────────────────────────────────────────────────────────
    # VxTQ family: GDSVTQ (8.2%), AQSVTQ (8.0%), AQKVTQ (4.1%), AQTVTQ (4.2%)
    r'[AG][DNQ][SKTVRGN]V[TNVSA]Q',
    # VEQ/VKQ family: GENVEQ (1.5%), GEQVEQ (1.5%), GEKVEQ (1.3%), QKEVEQ (1.8%)
    r'[GQKEI][QEKLMNIVDRSG][QKEVDLNIVGSA]V[EKD]Q',
    # VQQ family: GQQVQQ, QQKVQQ
    r'[GQK][QKE][QKVLN]V[QK]Q',
    # LxQ family: QQLEQS (1.0%), QELEQS (0.7%), QQLEQSP
    r'[QE][QKE]L[NQKE][QS]',
    # AKT family: IDAKTT (1.0%), SLAKTT (0.8%), GDAKTT (0.8%)
    r'[IGSDV][DLSAV][AGS]K[TS][TQ]',
    # IHQ/IKQ family: QQIKHF, SQKIEQ
    r'[QS][QK][IK][KHE][QHF]',
    # ── TRBV ────────────────────────────────────────────────────────────────
    # xGV main: DAGVTQ (4.4%), NAGVTQ (4.3%), DSGVTQ (4.1%), GAGVSQ (3.5%)
    # pos4=[VIA] (not L/F) to avoid false matches on IGHV internal sequences
    r'[DNEAGHSK][ADSGEP][GA][VIA][TISQAV]Q',
    # DAx: DAGITQ (2.0%), DAGVIQ (1.8%), DGGITQ (1.2%), DAEITQ (0.8%)
    r'D[ADGEPSVK][GAEVQKRDP][VI][TISQF]Q',
    # DTxV: DTEVTQ (2.6%), DTGVTQ (2.2%), DTAVSQ (1.4%), DTGITQ (1.0%)
    r'DT[EGAKD][VI][TSFIQ]Q',
    # EAx: EAEVTQ (1.2%), DSEVTQ (1.3%), DAEVTQ (1.2%), HAEVTQ, NADVTQ
    r'[ENDHSAG][ASDHE][EAQKDGT][VI][TS]Q',
    # GAL: GALVSQ (1.6%), GAMVIQ (0.6%), GALLSQ
    r'GA[LMV][VLI][TSIQ]Q',
    # EPE/AGV: EPEVTQ (1.4%), AGVVSQ
    r'[EA][PGS][EA][VI][TSIQ]Q',
    # AQT/SQT: AQTIHQ (1.1%), SQTIHQ
    r'[AS]QT[ILV][HNQ]',
    # DVK: DVKVTQ (0.6%), DVMVIQ
    r'D[VA][KRM][VI][TS]Q',
    # ── TRAV additional patterns (v2.2.0) — derived from training set + IMGT ──
    r'GQ[SNK][LIV][EQD]Q',   # TRAV1-1, TRAV1-2
    r'KDQ[VI][FY]Q',          # TRAV2
    r'[ED]NQ[VK][EQH]',       # TRAV7
    r'R[KNQ]EV[EK]',          # TRAV12-1
    r'GES[VT][GL]',            # TRAV13-2  (H/Q excluded: 36 IGHV FP)
    r'AQK[IV][TI]Q',          # TRAV14/DV4
    r'[SD]QQ[EGK][EQ]',       # TRAV17
    r'KQE[VK][TQ]Q',          # TRAV21
    r'GQQ[VK][MQVK]Q',        # TRAV25
    r'DQQV[KR]Q',             # TRAV29/DV5
    r'EDKV[VIMQ]',             # TRAV36/DV7 (V-only pos3: 1 IGHV FP elim.)
    r'SNSV[KR]Q',             # TRAV40
]


def find_vgene_start(sequence: str) -> int:
    """
    Find the probable start of V-gene using Framework 1 motifs.

    Returns:
        Position where V-gene likely starts (0 if no pattern found)
    """
    for pattern in VGENE_START_PATTERNS:
        match = re.search(pattern, sequence)
        if match:
            return match.start()
    return 0


def clean_sequence(
    sequence: str,
    max_length: int = 120,
    clean_terminals: bool = True
) -> Optional[str]:
    """
    Clean a protein sequence by trimming N and C terminals.

    Args:
        sequence: Raw protein sequence
        max_length: Maximum expected V-gene length
        clean_terminals: Whether to apply terminal cleaning

    Returns:
        Cleaned sequence or None if too short after cleaning
    """
    if not clean_terminals:
        return sequence

    # Clean N-terminal
    start_pos = find_vgene_start(sequence)
    cleaned = sequence[start_pos:]

    # Clean C-terminal (trim to typical V-gene length)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    # Check minimum length after cleaning
    if len(cleaned) < 70:
        return None

    return cleaned


def parse_tblastn_results(
    results_file: Path,
    min_length: int = 80,
    min_identity: float = 80.0
) -> List[Tuple[str, str, int, int, str, float, float]]:
    """
    Parse TBLASTN results with quality filters.

    Args:
        results_file: TBLASTN output file
        min_length: Minimum alignment length
        min_identity: Minimum percent identity (default: 80%)

    Returns list of (query_id, contig_id, start, end, strand, identity, evalue)
    """
    hits = []
    total_hits = 0
    filtered_identity = 0
    filtered_length = 0

    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue

            total_hits += 1
            fields = line.strip().split('\t')

            # Format: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen
            query_id = fields[0]
            contig_id = fields[1]
            pident = float(fields[2])
            length = int(fields[3])
            sstart = int(fields[8])
            send = int(fields[9])
            evalue = float(fields[10])

            # Filter by identity
            if pident < min_identity:
                filtered_identity += 1
                continue

            # Skip short hits
            if length < min_length:
                filtered_length += 1
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

            hits.append((query_id, contig_id, start, end, strand, pident, evalue))

    print(f"   Total TBLASTN hits: {total_hits:,}")
    print(f"   Filtered by identity (<{min_identity}%): {filtered_identity:,}")
    print(f"   Filtered by length (<{min_length} aa): {filtered_length:,}")
    print(f"   High-quality hits retained: {len(hits):,}")

    return hits


def merge_overlapping_hits(
    hits: List[Tuple[str, str, int, int, str, float, float]]
) -> List[Tuple[str, str, int, int, str]]:
    """
    Merge overlapping hits on the same contig and strand.

    Returns merged list of (query_id, contig_id, start, end, strand)
    """
    # Group by contig and strand
    grouped = defaultdict(list)
    for query_id, contig_id, start, end, strand, identity, evalue in hits:
        grouped[(contig_id, strand)].append((query_id, start, end, identity, evalue))

    merged = []

    for (contig_id, strand), coords in grouped.items():
        # Sort by start position
        coords.sort(key=lambda x: x[1])

        # Keep best hit per region (highest identity)
        current_query, current_start, current_end, best_identity, min_evalue = coords[0]

        for query_id, start, end, identity, evalue in coords[1:]:
            if start <= current_end + 50:  # Allow 50 bp gap
                current_end = max(current_end, end)
                if identity > best_identity:
                    current_query = query_id
                    best_identity = identity
                min_evalue = min(min_evalue, evalue)
            else:
                merged.append((current_query, contig_id, current_start, current_end, strand))
                current_query, current_start, current_end, best_identity, min_evalue = query_id, start, end, identity, evalue

        merged.append((current_query, contig_id, current_start, current_end, strand))

    return merged


def extract_sequences(
    genome_file: Path,
    regions: List[Tuple[str, str, int, int, str]],
    extend: int = 150,
    clean_terminals: bool = True,
    max_vgene_length: int = 120
) -> List[SeqRecord]:
    """
    Extract and translate sequences from genome.

    Args:
        genome_file: Genome FASTA file
        regions: List of (query_id, contig_id, start, end, strand)
        extend: Bases to extend on each side
        clean_terminals: Whether to clean N/C terminals
        max_vgene_length: Maximum expected V-gene length

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
    if clean_terminals:
        print(f"   🧹 Terminal cleaning: ENABLED (max length: {max_vgene_length} aa)")
    else:
        print(f"   ⚠️  Terminal cleaning: DISABLED")

    candidates = []
    failed = 0
    cleaned_count = 0
    n_trimmed_total = 0
    c_trimmed_total = 0

    for idx, (query_id, contig_id, start, end, strand) in enumerate(regions):
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
            original_length = len(best_protein)

            # Clean terminals
            cleaned_protein = clean_sequence(
                best_protein,
                max_length=max_vgene_length,
                clean_terminals=clean_terminals
            )

            if cleaned_protein:
                # Track trimming
                n_trim = find_vgene_start(best_protein)
                c_trim = max(0, len(cleaned_protein) - max_vgene_length)

                if n_trim > 0 or c_trim > 0:
                    cleaned_count += 1
                    n_trimmed_total += n_trim
                    c_trimmed_total += c_trim

                record = SeqRecord(
                    Seq(cleaned_protein),
                    id=f"candidate_{idx+1}",
                    description=f"query={query_id} {contig_id}:{extended_start}-{extended_end}({strand}) len={len(cleaned_protein)}"
                )
                candidates.append(record)
            else:
                failed += 1
        else:
            failed += 1

        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx+1}/{len(regions)}... (extracted: {len(candidates)}, failed: {failed})")

    print(f"   ✅ Extracted {len(candidates)} valid candidates (failed: {failed})")

    if clean_terminals and cleaned_count > 0:
        print(f"\n   🧹 Cleaning stats:")
        print(f"      Sequences cleaned: {cleaned_count}/{len(candidates)} ({cleaned_count/len(candidates)*100:.1f}%)")
        print(f"      Avg N-terminal trimmed: {n_trimmed_total/cleaned_count:.1f} aa")
        print(f"      Avg C-terminal trimmed: {c_trimmed_total/cleaned_count:.1f} aa")

    return candidates


def remove_duplicates(sequences: List[SeqRecord]) -> List[SeqRecord]:
    """Remove duplicate sequences by (query_id, sequence) to preserve all genes."""
    print(f"\n🔍 Removing duplicates (by query+sequence)...")

    seen = set()
    unique = []

    for rec in sequences:
        seq_str = str(rec.seq)

        # Extract query_id from description
        query_id = None
        if 'query=' in rec.description:
            query_id = rec.description.split('query=')[1].split()[0]

        # Use (query_id, sequence) as key
        key = (query_id, seq_str) if query_id else seq_str

        if key not in seen:
            seen.add(key)
            unique.append(rec)

    print(f"   Original: {len(sequences)}")
    print(f"   Unique: {len(unique)}")
    print(f"   Removed: {len(sequences) - len(unique)}")

    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Extract candidate V-gene sequences from TBLASTN results (v2.0 with auto-cleaning)"
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
    parser.add_argument(
        "--clean-terminals",
        action="store_true",
        help="Enable automatic N/C terminal cleaning (recommended)"
    )
    parser.add_argument(
        "--max-vgene-length",
        type=int,
        default=120,
        help="Maximum V-gene length for C-terminal trimming (default: 120)"
    )
    parser.add_argument(
        "--min-identity",
        type=float,
        default=80.0,
        help="Minimum percent identity for TBLASTN hits (default: 80.0)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging of overlapping hits (extract each hit separately)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EXTRACT V-GENE CANDIDATES - v2.0")
    print("=" * 70)

    # Parse TBLASTN results
    print(f"\n📊 Parsing TBLASTN results...")
    print(f"   File: {args.tblastn_results}")

    hits = parse_tblastn_results(args.tblastn_results, args.min_length, args.min_identity)

    # Merge overlapping hits (optional)
    if args.no_merge:
        print(f"\n🔗 Merge DISABLED - extracting each hit separately...")
        # Convert hits to regions without merging
        merged = [(query, contig, start, end, strand) for query, contig, start, end, strand, identity, evalue in hits]
        print(f"   ✅ Total regions to extract: {len(merged)}")
    else:
        print(f"\n🔗 Merging overlapping hits...")
        merged = merge_overlapping_hits(hits)
        print(f"   ✅ Merged to {len(merged)} regions")

    # Extract sequences
    candidates = extract_sequences(
        args.genome,
        merged,
        args.extend,
        args.clean_terminals,
        args.max_vgene_length
    )

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
    print(f"  python scripts/07_classify_candidates.py \\")
    print(f"      --candidates {args.output} \\")
    print(f"      --model models/v2_hybrid/best_model.pt \\")
    print(f"      --output results/<species>/vgenes_predicted.fasta")


if __name__ == "__main__":
    main()
