#!/usr/bin/env python3
"""
Parse IMGT Protein displays to FASTA.
Generic parser for any species.
Format: metadata line followed by sequence line.
"""

import argparse
import re
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os

def parse_imgt_table(input_file, species_code=None):
    """
    Parse IMGT protein display table.
    Format: pairs of lines
    Line 1: [Species]  IGHV1-4  IGHV1-4*01  AC073561  VH  F
    Line 2: QVQLQQSGA.ELARP GASVKMSCKAS GYTF....TSYT...

    Args:
        input_file: Path to raw IMGT table
        species_code: Species prefix (e.g., 'Musmus', 'Homsap').
                     If None, auto-detect from first line.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Auto-detect species if not provided
    if species_code is None:
        for line in lines:
            line = line.strip()
            # Look for common species prefixes
            if line.startswith(('Musmus', 'Homsap', 'Ratrat', 'Galga', 'Sussc', 'Bosta', 'Musput')):
                species_code = line.split()[0]
                print(f"  Auto-detected species: {species_code}")
                break

        if species_code is None:
            print(f"  ⚠️  Could not auto-detect species in {input_file}")
            return []

    records = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for lines starting with species code
        if line.startswith(species_code):
            # Parse metadata line (tab or space separated)
            fields = re.split(r'\s+', line)

            if len(fields) < 6:
                i += 1
                continue

            species_gene = fields[0]   # Musmus/Homsap
            gene = fields[1]            # IGHV1-4
            allele = fields[2]          # IGHV1-4*01
            accnum = fields[3]          # AC073561
            domain = fields[4]          # VH
            functionality = fields[5]   # F, P, ORF

            # Only functional
            if functionality != 'F':
                i += 1
                continue

            # Next line should be the sequence
            i += 1
            if i >= len(lines):
                break

            sequence_line = lines[i].strip()

            # Clean sequence: remove gaps (.) and spaces
            sequence = sequence_line.replace('.', '').replace(' ', '').strip()

            # Skip if too short
            if len(sequence) < 50:
                i += 1
                continue

            # Create record
            record = SeqRecord(
                Seq(sequence),
                id=allele,
                description=f"gene={gene} accnum={accnum} functionality={functionality}"
            )
            records.append(record)

        i += 1

    return records

def main():
    parser = argparse.ArgumentParser(
        description="Parse IMGT protein display tables to FASTA (generic for any species)"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with IMGT raw files (*_raw.txt)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for FASTA files")
    parser.add_argument("--species", default=None,
                        help="Species code (e.g., 'Musmus', 'Homsap'). Auto-detect if not provided.")
    parser.add_argument("--suffix", default="imgt",
                        help="Output file suffix (default: imgt)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each locus
    loci = ['ighv', 'igkv', 'iglv', 'trav', 'trbv']

    total = 0
    all_records = []
    for locus in loci:
        input_file = os.path.join(args.input_dir, f"{locus}_raw.txt")

        if not os.path.exists(input_file):
            continue  # Silently skip missing files (e.g., iglv for mouse)

        print(f"\nProcessing {locus.upper()}...")
        records = parse_imgt_table(input_file, species_code=args.species)

        if records:
            # Determine species name for output filename
            if args.species:
                species_name = args.species.lower()
            else:
                # Try to infer from first record or use generic
                species_name = "species"
                if records:
                    # Extract from first line of file
                    with open(input_file, 'r') as f:
                        first_line = f.readline()
                        if 'Musmus' in first_line:
                            species_name = 'mouse'
                        elif 'Homsap' in first_line:
                            species_name = 'human'
                        elif 'Ratrat' in first_line:
                            species_name = 'rat'
                        elif 'Galga' in first_line:
                            species_name = 'chicken'
                        elif 'Musput' in first_line:
                            species_name = 'ferret'

            output_file = os.path.join(args.output_dir, f"{locus}_{species_name}_{args.suffix}.fasta")
            SeqIO.write(records, output_file, 'fasta')
            print(f"  ✅ {len(records)} functional genes → {output_file}")
            total += len(records)
            all_records.extend(records)
        else:
            print(f"  ⚠️  No functional genes found")

    # Write combined file with all loci
    if all_records:
        combined_file = os.path.join(args.output_dir, "all_vgenes_imgt.fasta")
        SeqIO.write(all_records, combined_file, 'fasta')

    print(f"\n{'='*70}")
    print(f"TOTAL: {total} functional V-genes extracted from IMGT")
    if all_records:
        print(f"Combined: {combined_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
