#!/usr/bin/env python3
"""
Script 09: Phylogenetic Validation (Optional)
Phylogenetic validation:
1. Combine IMGT + predictions
2. MSA with ClustalO
3. Build tree (Nexus format)
4. Visualize with SeaView
"""

import argparse
import subprocess
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def combine_sequences(imgt_fasta, predictions_fasta, predictions_csv, output_fasta, top_n=20):
    """
    Combine IMGT reference with top predictions.

    Args:
        imgt_fasta: IMGT reference sequences
        predictions_fasta: All predicted sequences
        predictions_csv: CSV with probabilities
        output_fasta: Output combined FASTA
        top_n: Number of top predictions per locus to include
    """

    import pandas as pd

    print(f"\nCombining sequences...")

    # Load predictions with metadata
    pred_df = pd.read_csv(predictions_csv)
    vgenes = pred_df[pred_df['predicted_locus'] != 'background'].copy()

    # Get top N per locus
    top_predictions = []
    for locus in ['IGHV', 'IGKV', 'TRAV', 'TRBV']:
        locus_df = vgenes[vgenes['predicted_locus'] == locus].nlargest(top_n, 'probability')
        top_predictions.extend(locus_df['id'].tolist())

    print(f"  Selected {len(top_predictions)} top predictions")

    # Load prediction sequences
    pred_seqs = {rec.id: rec for rec in SeqIO.parse(predictions_fasta, 'fasta')}

    # Combine: IMGT + top predictions
    combined = []

    # Add IMGT sequences with prefix
    imgt_count = 0
    for rec in SeqIO.parse(imgt_fasta, 'fasta'):
        new_rec = SeqRecord(
            rec.seq,
            id=f"IMGT_{rec.id}",
            description=rec.description
        )
        combined.append(new_rec)
        imgt_count += 1

    print(f"  Added {imgt_count} IMGT sequences")

    # Add top predictions with prefix
    pred_count = 0
    for pred_id in top_predictions:
        if pred_id in pred_seqs:
            rec = pred_seqs[pred_id]
            # Get predicted locus
            locus = vgenes[vgenes['id'] == pred_id]['predicted_locus'].values[0]
            prob = vgenes[vgenes['id'] == pred_id]['probability'].values[0]

            new_rec = SeqRecord(
                rec.seq,
                id=f"PRED_{locus}_{pred_id}",
                description=f"prob={prob:.4f}"
            )
            combined.append(new_rec)
            pred_count += 1

    print(f"  Added {pred_count} prediction sequences")

    # Write combined
    SeqIO.write(combined, output_fasta, 'fasta')
    print(f"  ✅ Combined FASTA: {output_fasta}")
    print(f"     Total sequences: {len(combined)}")

    return output_fasta

def run_clustalo(input_fasta, output_aln, threads=8):
    """Run ClustalO multiple sequence alignment."""

    print(f"\nRunning ClustalO alignment...")
    print(f"  Input: {input_fasta}")
    print(f"  This may take several minutes...")

    cmd = [
        'clustalo',
        '-i', input_fasta,
        '-o', output_aln,
        '--outfmt=fasta',
        '--threads', str(threads),
        '--force'
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✅ Alignment complete: {output_aln}")
        return output_aln
    except subprocess.CalledProcessError as e:
        print(f"  ❌ ClustalO failed: {e.stderr}")
        return None

def build_tree(alignment_fasta, output_tree):
    """Build phylogenetic tree with ClustalO."""

    print(f"\nBuilding phylogenetic tree...")

    cmd = [
        'clustalo',
        '-i', alignment_fasta,
        '--guidetree-out', output_tree,
        '--force'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✅ Tree generated: {output_tree}")
        return output_tree
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Tree building failed: {e.stderr}")
        return None

def convert_to_nexus(newick_file, nexus_file):
    """Convert Newick tree to Nexus format for SeaView."""

    print(f"\nConverting to Nexus format...")

    try:
        from Bio import Phylo

        # Read Newick
        tree = Phylo.read(newick_file, 'newick')

        # Write Nexus
        Phylo.write(tree, nexus_file, 'nexus')

        print(f"  ✅ Nexus tree: {nexus_file}")
        return nexus_file
    except Exception as e:
        print(f"  ⚠️  Conversion failed: {e}")
        print(f"     You can open {newick_file} directly in SeaView")
        return newick_file

def main():
    parser = argparse.ArgumentParser(
        description="Phylogenetic validation with ClustalO + SeaView"
    )
    parser.add_argument("--imgt", required=True,
                        help="IMGT reference FASTA")
    parser.add_argument("--predictions", required=True,
                        help="Predicted V-genes FASTA")
    parser.add_argument("--predictions-csv", required=True,
                        help="Predictions CSV with probabilities")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top N predictions per locus (default: 20)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads for ClustalO")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("PHYLOGENETIC VALIDATION")
    print("="*70)

    # File paths
    combined_fasta = os.path.join(args.output_dir, "combined_sequences.fasta")
    alignment_fasta = os.path.join(args.output_dir, "alignment.fasta")
    tree_newick = os.path.join(args.output_dir, "tree.newick")
    tree_nexus = os.path.join(args.output_dir, "tree.nexus")

    # Step 1: Combine sequences
    combine_sequences(
        args.imgt,
        args.predictions,
        args.predictions_csv,
        combined_fasta,
        args.top_n
    )

    # Step 2: Multiple sequence alignment
    alignment = run_clustalo(combined_fasta, alignment_fasta, args.threads)

    if not alignment:
        print("\n❌ Alignment failed. Cannot proceed.")
        return

    # Step 3: Build tree
    tree = build_tree(alignment_fasta, tree_newick)

    if not tree:
        print("\n❌ Tree building failed.")
        return

    # Step 4: Convert to Nexus
    convert_to_nexus(tree_newick, tree_nexus)

    print("\n" + "="*70)
    print("PHYLOGENETIC ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print(f"  1. Open SeaView:")
    print(f"     seaview {tree_nexus}")
    print(f"\n  2. Or open alignment + tree in SeaView:")
    print(f"     seaview {alignment_fasta}")
    print(f"\n  3. Look for predictions clustering with IMGT sequences")
    print(f"     - Same branch = identical/very similar")
    print(f"     - Predictions far from all IMGT = potential false positives")
    print("="*70)

if __name__ == "__main__":
    main()
