#!/usr/bin/env python3
"""
parse_imgt_display.py — Parse IMGT protein display format to clean FASTA.

Input format (IMGT protein display, tab-separated header + aligned sequence):
    Oncmyk  TRBV1S1  TRBV1S1*01  U18123  V-BETA  F
    GSGVTQ.TPTI LWEL..KDAA IQWYQQEPA GSVTLL.. ...

Usage:
    python parse_imgt_display.py input.txt output.fasta
    python parse_imgt_display.py input.txt output.fasta --locus TRBV
"""

import sys
import re
import argparse
from pathlib import Path

# Standard amino acid alphabet (no ambiguous/non-standard characters)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
NON_STANDARD = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

# Column indices in IMGT header (0-based, tab-separated)
COL_SPECIES      = 0
COL_GENE         = 1
COL_ALLELE       = 2
COL_ACCNUM       = 3
COL_DOMAIN_LABEL = 4
COL_FUNCTIONAL   = 5   # F / P / ORF


def is_header_line(line: str) -> bool:
    """Detect a gene entry header line (tab-separated, >= 5 fields, last field is F/P/ORF)."""
    parts = line.rstrip("\n").split("\t")
    # Must have at least 6 tab-separated fields
    if len(parts) < 6:
        return False
    # Field 0: short species code (no spaces, no dots, typically 6 chars like Oncmyk)
    if " " in parts[0] or "." in parts[0] or len(parts[0]) == 0:
        return False
    # Field 5: functionality must be F, P, or ORF
    func = parts[5].strip()
    return func in ("F", "P", "ORF")


def clean_sequence(raw_seq: str) -> str:
    """Remove IMGT gap characters (dots), spaces, and newlines from aligned sequence."""
    return re.sub(r"[\s.]", "", raw_seq).upper()


def parse_imgt_display(filepath: str, functional_only: bool = True) -> list[dict]:
    """
    Parse IMGT protein display file.

    Returns list of dicts with keys:
        species_code, gene, allele, accnum, functionality, sequence
    """
    records = []
    current_header = None

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.rstrip("\n")

            # Skip blank lines
            if not line.strip():
                current_header = None
                continue

            if is_header_line(line):
                parts = line.split("\t")
                func = parts[COL_FUNCTIONAL].strip()

                if functional_only and func != "F":
                    current_header = None   # skip; next line is its sequence
                    continue

                current_header = {
                    "species_code": parts[COL_SPECIES].strip(),
                    "gene":         parts[COL_GENE].strip(),
                    "allele":       parts[COL_ALLELE].strip(),
                    "accnum":       parts[COL_ACCNUM].strip(),
                    "functionality": func,
                }

            elif current_header is not None:
                # This line is the sequence for the preceding header
                seq = clean_sequence(line)

                if len(seq) < 50:
                    # Too short — probably a preamble/annotation line that slipped through
                    current_header = None
                    continue

                # Check for non-standard amino acids
                bad = NON_STANDARD.findall(seq)
                if bad:
                    bad_chars = sorted(set(bad))
                    print(
                        f"  [SKIP] {current_header['allele']} — "
                        f"non-standard characters: {bad_chars}",
                        file=sys.stderr,
                    )
                    current_header = None
                    continue

                record = dict(current_header)
                record["sequence"] = seq
                records.append(record)
                current_header = None   # consumed

    return records


def write_fasta(records: list[dict], outpath: str, locus: str | None = None) -> None:
    """Write records to FASTA. Header: >SpeciesCode_Allele"""
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    with open(outpath, "w", encoding="utf-8") as fh:
        for rec in records:
            # Optional locus tag in header for traceability
            locus_tag = f" locus={locus}" if locus else ""
            header = f">{rec['species_code']}_{rec['allele']}{locus_tag} accnum={rec['accnum']} functionality={rec['functionality']}"
            fh.write(header + "\n")
            # Wrap sequence at 70 chars (standard FASTA)
            seq = rec["sequence"]
            for i in range(0, len(seq), 70):
                fh.write(seq[i : i + 70] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse IMGT protein display format to clean FASTA."
    )
    parser.add_argument("input",  help="Input file (IMGT protein display .txt)")
    parser.add_argument("output", help="Output FASTA file path")
    parser.add_argument(
        "--locus", default=None,
        help="Locus tag to embed in FASTA headers (e.g. TRBV, IGHV). Optional."
    )
    parser.add_argument(
        "--all", dest="all_functional", action="store_true", default=False,
        help="Include P (pseudogene) and ORF entries in addition to F (functional). "
             "Default: functional only."
    )
    args = parser.parse_args()

    functional_only = not args.all_functional

    print(f"Parsing: {args.input}", file=sys.stderr)
    print(f"Functional only: {functional_only}", file=sys.stderr)

    records = parse_imgt_display(args.input, functional_only=functional_only)

    if not records:
        print("ERROR: No records parsed. Check input format.", file=sys.stderr)
        sys.exit(1)

    # Summary
    species_counts: dict[str, int] = {}
    for r in records:
        species_counts[r["species_code"]] = species_counts.get(r["species_code"], 0) + 1

    print(f"\nParsed {len(records)} sequences:", file=sys.stderr)
    for sp, n in sorted(species_counts.items()):
        print(f"  {sp}: {n}", file=sys.stderr)

    write_fasta(records, args.output, locus=args.locus)
    print(f"\nOutput written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
