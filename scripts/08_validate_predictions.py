#!/usr/bin/env python3
"""
Script 08: Validate Predictions Against Reference 
Validate predictions against reference V-genes using BLASTP.
Works with any reference: NCBI, IMGT, etc.
"""

import os
import argparse
import subprocess
import pandas as pd
from collections import defaultdict

def run_blastp(query, reference, output, evalue=1e-10):
    """Run BLASTP predictions vs reference."""

    print(f"\nRunning BLASTP...")
    print(f"  Query: {query}")
    print(f"  Reference: {reference}")

    # Create BLAST database
    db_path = reference.replace('.fasta', '_db')

    cmd_makedb = [
        'makeblastdb',
        '-in', reference,
        '-dbtype', 'prot',
        '-out', db_path
    ]

    print("  Creating BLAST database...")
    subprocess.run(cmd_makedb, check=True, capture_output=True)

    # Run BLASTP
    cmd_blast = [
        'blastp',
        '-query', query,
        '-db', db_path,
        '-out', output,
        '-outfmt', '6 qseqid sseqid pident length qlen slen qcovs evalue bitscore',
        '-evalue', str(evalue),
        '-max_target_seqs', '5',
        '-num_threads', '8'
    ]

    print("  Running BLASTP...")
    result = subprocess.run(cmd_blast, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ⚠️  BLASTP stderr: {result.stderr}")

    print(f"  ✅ Results saved to: {output}")

    return output

def parse_blast_results(blast_file, min_identity=95, min_coverage=90):
    """Parse BLAST results and filter by quality."""

    if not os.path.exists(blast_file) or os.path.getsize(blast_file) == 0:
        print("⚠️  No BLAST hits found")
        return pd.DataFrame(), []

    # Load BLAST results
    cols = ['qseqid', 'sseqid', 'pident', 'length', 'qlen', 'slen', 'qcovs', 'evalue', 'bitscore']
    blast_df = pd.read_csv(blast_file, sep='\t', names=cols)

    print(f"\nTotal BLAST hits: {len(blast_df)}")

    # Filter by identity and coverage
    high_quality = blast_df[
        (blast_df['pident'] >= min_identity) &
        (blast_df['qcovs'] >= min_coverage)
    ]

    print(f"High-quality hits (≥{min_identity}% identity, ≥{min_coverage}% coverage): {len(high_quality)}")

    # Get best hit per query
    best_hits = high_quality.sort_values('bitscore', ascending=False).groupby('qseqid').first().reset_index()

    return blast_df, best_hits

def extract_locus(gene_name):
    """Extract locus from gene name."""
    gene_upper = gene_name.upper()
    if 'IGHV' in gene_upper:
        return 'IGHV'
    elif 'IGKV' in gene_upper or 'IGLV' in gene_upper:
        return 'IGKV'
    elif 'TRAV' in gene_upper:
        return 'TRAV'
    elif 'TRBV' in gene_upper:
        return 'TRBV'
    elif 'TRDV' in gene_upper:
        return 'TRDV'
    elif 'TRGV' in gene_upper:
        return 'TRGV'
    return 'unknown'

def validate_predictions(predictions_csv, best_hits):
    """Validate predictions against reference."""

    # Load predictions
    pred_df = pd.read_csv(predictions_csv)

    # Filter V-genes only (no background)
    vgenes = pred_df[pred_df['predicted_locus'] != 'background'].copy()

    print(f"\nTotal V-gene predictions: {len(vgenes)}")

    # Extract query IDs from predictions to count unique IMGT genes
    from Bio import SeqIO
    print("  Extracting unique IMGT genes from predictions...")
    query_to_imgt = {}
    # Note: predictions CSV doesn't have query IDs, we need to parse FASTA
    # This will be passed from main()

    # Merge with BLAST hits
    vgenes_matched = vgenes.merge(
        best_hits[['qseqid', 'sseqid', 'pident', 'qcovs']],
        left_on='id',
        right_on='qseqid',
        how='left'
    )

    # Extract reference locus
    vgenes_matched['ref_locus'] = vgenes_matched['sseqid'].apply(
        lambda x: extract_locus(x) if pd.notna(x) else None
    )

    # Categorize
    validated = vgenes_matched[
        (vgenes_matched['predicted_locus'] == vgenes_matched['ref_locus'])
    ]

    misclassified = vgenes_matched[
        (vgenes_matched['ref_locus'].notna()) &
        (vgenes_matched['predicted_locus'] != vgenes_matched['ref_locus'])
    ]

    no_match = vgenes_matched[vgenes_matched['ref_locus'].isna()]

    return vgenes_matched, validated, misclassified, no_match


def print_report(pred_df, validated, misclassified, no_match, ref_counts, predictions_fasta):
    """Print validation report with REAL recall using query IDs from FASTA."""
    from Bio import SeqIO

    print("\n" + "="*70)
    print("VALIDATION REPORT - IMGT REFERENCE")
    print("="*70)

    # Extract query IDs from FASTA
    print("  Extracting query IDs from predictions FASTA...")
    query_to_imgt = {}
    for rec in SeqIO.parse(predictions_fasta, 'fasta'):
        if 'query=' in rec.description:
            query_id = rec.description.split('query=')[1].split()[0]
            query_to_imgt[rec.id] = query_id

    # Add query IDs to dataframes
    validated_with_query = validated.copy()
    validated_with_query['imgt_gene'] = validated_with_query['id'].map(query_to_imgt)

    total_predictions = len(pred_df)
    unique_genes_found = len(set(validated_with_query['imgt_gene'].dropna()))

    print(f"\nTotal V-gene predictions: {total_predictions}")
    print(f"  ✅ Validated (correct locus): {len(validated)} ({len(validated)/total_predictions*100:.1f}%)")
    print(f"  ❌ Misclassified (wrong locus): {len(misclassified)} ({len(misclassified)/total_predictions*100:.1f}%)")
    print(f"  ⚠️  No IMGT match: {len(no_match)} ({len(no_match)/total_predictions*100:.1f}%)")

    print(f"\nUnique IMGT genes found: {unique_genes_found}")

    # Precision (of those with matches)
    with_match = len(validated) + len(misclassified)
    if with_match > 0:
        precision = len(validated) / with_match * 100
        print(f"Precision (of matched predictions): {precision:.1f}%")

    # Per-locus breakdown with UNIQUE genes
    print("\n" + "-"*80)
    print("PER-LOCUS BREAKDOWN (UNIQUE IMGT GENES):")
    print("-"*80)
    print(f"{'Locus':<10} {'Predictions':<12} {'Unique Found':<15} {'Total IMGT':<12} {'Recall':<10} {'Precision':<12}")
    print("-"*80)

    total_found = 0
    for locus in ['IGHV', 'IGKV', 'TRAV', 'TRBV']:
        n_pred = len(pred_df[pred_df['predicted_locus'] == locus])

        # Count UNIQUE IMGT genes found for this locus
        locus_validated = validated_with_query[validated_with_query['predicted_locus'] == locus]
        unique_found = len(set(locus_validated['imgt_gene'].dropna()))
        total_found += unique_found

        # Calculate precision for this locus
        locus_all = pred_df[pred_df['predicted_locus'] == locus]
        locus_correct = validated[validated['predicted_locus'] == locus]
        locus_precision = (len(locus_correct) / len(locus_all) * 100) if len(locus_all) > 0 else 0

        n_imgt = ref_counts.get(locus, 0)
        recall = (unique_found / n_imgt * 100) if n_imgt > 0 else 0

        print(f"{locus:<10} {n_pred:<12} {unique_found:<15} {n_imgt:<12} {recall:<10.1f} {locus_precision:<11.1f}")
    # Total row
    total_imgt = sum(ref_counts.values())
    total_recall = (total_found / total_imgt * 100) if total_imgt > 0 else 0
    total_precision = (len(validated) / total_predictions * 100) if total_predictions > 0 else 0
    print("-"*80)
    print(f"{'TOTAL':<10} {total_predictions:<12} {total_found:<15} {total_imgt:<12} {total_recall:<10.1f} {total_precision:<11.1f}")

    # Misclassification examples
    if len(misclassified) > 0:
        print("\n" + "-"*70)
        print("MISCLASSIFICATION EXAMPLES:")
        print("-"*80)
        for _, row in misclassified.head(10).iterrows():
            print(f"  {row['id'][:30]:30s} Predicted: {row['predicted_locus']:5s} → "
                  f"IMGT: {row['ref_locus']:5s} ({row['pident']:.1f}% identity)")


def main():
    parser = argparse.ArgumentParser(description="Validate predictions against reference")
    parser.add_argument("--predictions", required=True, help="Predicted V-genes FASTA")
    parser.add_argument("--predictions-csv", required=True, help="Predictions CSV")
    parser.add_argument("--reference", required=True, help="Reference FASTA (IMGT/NCBI)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-identity", type=float, default=80.0,
                        help="Minimum percent identity")
    parser.add_argument("--min-coverage", type=float, default=70.0,
                        help="Minimum query coverage")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("VALIDATION AGAINST REFERENCE")
    print("="*70)
    print(f"Reference: {args.reference}")
    print(f"Criteria: ≥{args.min_identity}% identity, ≥{args.min_coverage}% coverage")

    # BLASTP
    blast_output = os.path.join(args.output_dir, "blast_results.txt")
    run_blastp(args.predictions, args.reference, blast_output)

    # Parse
    blast_df, best_hits = parse_blast_results(
        blast_output,
        args.min_identity,
        args.min_coverage
    )

    if len(best_hits) == 0:
        print("\n❌ No high-quality matches found. Try lowering thresholds.")
        return

    # Validate
    pred_matched, validated, misclassified, no_match = validate_predictions(
        args.predictions_csv,
        best_hits
    )

    # Reference counts
    from Bio import SeqIO
    ref_seqs = list(SeqIO.parse(args.reference, 'fasta'))
    ref_counts = defaultdict(int)
    for rec in ref_seqs:
        locus = extract_locus(rec.id)
        if locus in ['IGHV', 'IGKV', 'TRAV', 'TRBV']:
            ref_counts[locus] += 1

    # Save results
    validated.to_csv(os.path.join(args.output_dir, 'validated.csv'), index=False)
    misclassified.to_csv(os.path.join(args.output_dir, 'misclassified.csv'), index=False)
    no_match.to_csv(os.path.join(args.output_dir, 'no_match.csv'), index=False)

    print(f"\n✅ Saved results to: {args.output_dir}")

    # Print report
    vgenes_only = pred_matched[pred_matched['predicted_locus'] != 'background']
    print_report(vgenes_only, validated, misclassified, no_match, ref_counts, args.predictions)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
