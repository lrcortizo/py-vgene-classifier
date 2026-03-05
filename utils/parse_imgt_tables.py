#!/usr/bin/env python3
"""
Parse IMGT Protein displays to FASTA.
Format: metadata line followed by sequence line.
"""

import argparse
import re
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os

def parse_imgt_table(input_file):
    """
    Parse IMGT protein display table.
    Format: pairs of lines
    Line 1: Musmus  IGHV1-4  IGHV1-4*01  AC073561  VH  F
    Line 2: QVQLQQSGA.ELARP GASVKMSCKAS GYTF....TSYT...
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for lines starting with "Musmus"
        if line.startswith('Musmus'):
            # Parse metadata line
            fields = line.split()

            if len(fields) < 6:
                i += 1
                continue

            species_gene = fields[0]  # Musmus
            gene = fields[1]           # IGHV1-4
            allele = fields[2]         # IGHV1-4*01
            accnum = fields[3]         # AC073561
            domain = fields[4]         # VH
            functionality = fields[5]  # F, P, ORF

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
        description="Parse IMGT protein display tables to FASTA"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with IMGT raw files (*_raw.txt)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for FASTA files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each locus
    loci = ['ighv', 'igkv', 'trav', 'trbv']

    total = 0
    for locus in loci:
        input_file = os.path.join(args.input_dir, f"{locus}_raw.txt")
        output_file = os.path.join(args.output_dir, f"{locus}_mouse_imgt.fasta")

        if not os.path.exists(input_file):
            print(f"⚠️  {input_file} not found, skipping")
            continue

        print(f"\nProcessing {locus.upper()}...")
        records = parse_imgt_table(input_file)

        if records:
            SeqIO.write(records, output_file, 'fasta')
            print(f"  ✅ {len(records)} functional genes → {output_file}")
            total += len(records)
        else:
            print(f"  ⚠️  No functional genes found")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total} functional V-genes extracted from IMGT")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
