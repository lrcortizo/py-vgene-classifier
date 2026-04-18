#!/usr/bin/env python3
"""
Extract V-genes from aligned multi-species FASTA files.
Converts alignment format with gaps to clean FASTA sequences.

Usage:
    python utils/extract_multispecies_vgenes.py \
        --input-dir /path/to/alignment_results \
        --output-dir data/raw/multispecies \
        --taxa mammals \
        --min-length 60
"""

import argparse
from pathlib import Path


def clean_sequence(seq):
    """Remove gaps and clean sequence."""
    seq = seq.replace('-', '').replace('.', '')
    seq = ''.join(seq.split())
    return seq.upper()


def parse_header(header):
    """
    Parse multi-species alignment header:
    >Vs316|Condylura_cristata-AJFV0 FR1:1-26...
    
    Returns: (vs_id, species)
    """
    header = header.lstrip('>')
    parts = header.split('|')
    if len(parts) < 2:
        return None, None
    
    vs_id = parts[0].strip()
    species_part = parts[1].split()[0]
    species = species_part.split('-')[0]
    
    return vs_id, species


def extract_vgenes(input_file, output_file, locus, min_length):
    """Extract V-genes from alignment format to simple FASTA."""
    
    sequences = {}
    current_id = None
    current_species = None
    current_seq = []
    
    print(f"\n{'='*70}")
    print(f"Processing {locus.upper()}")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('>'):
                # Save previous sequence
                if current_id and current_seq:
                    seq = clean_sequence(''.join(current_seq))
                    if len(seq) >= min_length:
                        key = f"{current_species}_{current_id}"
                        sequences[key] = seq
                
                # Parse new header
                vs_id, species = parse_header(line)
                current_id = vs_id
                current_species = species
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_id and current_seq:
            seq = clean_sequence(''.join(current_seq))
            if len(seq) >= min_length:
                key = f"{current_species}_{current_id}"
                sequences[key] = seq
    
    # Count species
    species_set = set(k.split('_')[0] for k in sequences.keys())
    
    print(f"Extracted: {len(sequences)} sequences")
    print(f"Species:   {len(species_set)}")
    print(f"Output:    {output_file}")
    
    # Write output WITH locus= tag for compatibility with pipeline
    locus_upper = locus.upper()
    with open(output_file, 'w') as f:
        for seq_id, seq in sorted(sequences.items()):
            f.write(f">{seq_id} locus={locus_upper}\n")
            f.write(f"{seq}\n")
    
    return len(sequences), len(species_set)


def main():
    parser = argparse.ArgumentParser(
        description='Extract multi-species V-genes from alignment files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python utils/extract_multispecies_vgenes.py \\
        --input-dir /mnt/c/Users/luisr/PhD/alignment_to_imgt \\
        --output-dir data/raw/multispecies_mammals \\
        --taxa mammals \\
        --min-length 60
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing alignment results (e.g., resultsIGHV/, resultsTRAV/)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for extracted FASTA files'
    )
    
    parser.add_argument(
        '--taxa',
        choices=['mammals', 'reptiles'],
        default='mammals',
        help='Which taxa to extract (default: mammals)'
    )
    
    parser.add_argument(
        '--loci',
        nargs='+',
        default=['ighv', 'igkv', 'trav', 'trbv'],
        help='Loci to process (default: ighv igkv trav trbv)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=60,
        help='Minimum sequence length to keep (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Build file paths based on taxa
    loci_files = {}
    for locus in args.loci:
        locus_upper = locus.upper()
        filename = f"{locus}_{args.taxa}.realigned.annotated.fasta"
        input_path = args.input_dir / f"results{locus_upper}" / filename
        loci_files[locus] = input_path
    
    print("\n" + "="*70)
    print(f"EXTRACTING {args.taxa.upper()} V-GENES FROM ALIGNMENTS")
    print("="*70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Taxa:             {args.taxa}")
    print(f"Min length:       {args.min_length} aa")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    total_seqs = 0
    all_species = set()
    stats = {}
    
    for locus, input_file in loci_files.items():
        output_file = args.output_dir / f"{locus}.fasta"
        
        if not input_file.exists():
            print(f"\n⚠️  File not found: {input_file}")
            continue
        
        n_seqs, n_species = extract_vgenes(input_file, output_file, locus, args.min_length)
        total_seqs += n_seqs
        stats[locus] = (n_seqs, n_species)
        
        # Track species across loci
        with open(output_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    species = line.split('_')[0].lstrip('>')
                    all_species.add(species)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for locus, (n_seqs, n_sp) in stats.items():
        print(f"{locus.upper():5s}: {n_seqs:5d} sequences × {n_sp:3d} species")
    print("-"*70)
    print(f"TOTAL: {total_seqs:5d} sequences × {len(all_species):3d} unique species")
    print("="*70)
    print(f"\n✅ Output files saved to: {args.output_dir}/")
    
    # List output files
    output_files = sorted(args.output_dir.glob("*.fasta"))
    for f in output_files:
        print(f"   - {f.name}")
    print()


if __name__ == "__main__":
    main()
