"""
Filter CNN-positive candidates and prepare for validation
"""
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import sys

def filter_positives(predictions_csv, candidates_fasta, output_fasta, threshold=0.9):
    """Filter candidates predicted as V-genes by CNN"""
    
    # Load predictions
    df = pd.read_csv(predictions_csv)
    
    # Filter positives
    positives = df[df['probability'] >= threshold].copy()
    
    print(f"{'='*70}")
    print(f"FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Total candidates: {len(df)}")
    print(f"CNN positives (prob >= {threshold}): {len(positives)}")
    print(f"CNN negatives: {len(df) - len(positives)}")
    print(f"Success rate: {len(positives)/len(df)*100:.1f}%")
    
    # Load FASTA and filter
    candidates = SeqIO.to_dict(SeqIO.parse(candidates_fasta, "fasta"))
    
    positive_ids = set(positives['sequence_id'])
    
    # Save filtered sequences
    print(f"\n💾 Saving validated V-genes to {output_fasta}")
    with open(output_fasta, 'w') as f:
        for seq_id in positive_ids:
            if seq_id in candidates:
                SeqIO.write(candidates[seq_id], f, "fasta")
    
    print(f"   ✅ Saved {len(positive_ids)} sequences")
    
    # Save detailed summary
    summary_file = output_fasta.parent / f"{output_fasta.stem}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"V-gene Discovery Summary - Myotis lucifugus\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Pipeline Results:\n")
        f.write(f"  TBLASTN hits: 31,736\n")
        f.write(f"  After quality filter: 9,702\n")
        f.write(f"  Candidates extracted: {len(df)}\n")
        f.write(f"  CNN-validated V-genes: {len(positives)}\n")
        f.write(f"  Validation rate: {len(positives)/len(df)*100:.1f}%\n\n")
        
        f.write(f"Validated V-genes:\n")
        f.write(f"{'-'*70}\n")
        
        # Sort by probability
        positives_sorted = positives.sort_values('probability', ascending=False)
        
        for idx, row in positives_sorted.iterrows():
            f.write(f"\n{row['sequence_id']}\n")
            f.write(f"  Probability: {row['probability']:.4f}\n")
            f.write(f"  Length: {row['length']} aa\n")
            seq_short = row['sequence'][:60] + "..." if len(row['sequence']) > 60 else row['sequence']
            f.write(f"  Sequence: {seq_short}\n")
    
    print(f"📄 Summary saved to {summary_file}")
    
    # Show top candidates
    print(f"\n{'='*70}")
    print(f"TOP VALIDATED V-GENES")
    print(f"{'='*70}")
    
    for idx, row in positives_sorted.head(11).iterrows():
        print(f"\n{row['sequence_id']}")
        print(f"  Probability: {row['probability']:.4f}")
        print(f"  Length: {row['length']} aa")
        seq_short = row['sequence'][:50] + "..." if len(row['sequence']) > 50 else row['sequence']
        print(f"  Sequence: {seq_short}")
    
    return positives

if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).parent.parent
    RESULTS_DIR = PROJECT_DIR / "results"
    
    SPECIES = "myotis_lucifugus"
    SPECIES_RESULTS = RESULTS_DIR / SPECIES
    
    PREDICTIONS = SPECIES_RESULTS / "predictions.csv"
    CANDIDATES = SPECIES_RESULTS / "candidates.fasta"
    OUTPUT = SPECIES_RESULTS / "validated_vgenes.fasta"
    
    if not PREDICTIONS.exists():
        print(f"❌ Predictions file not found: {PREDICTIONS}")
        print(f"   Run predict_fasta.py first")
        sys.exit(1)
    
    print("="*70)
    print("FILTER CNN-VALIDATED V-GENES")
    print("="*70)
    
    positives = filter_positives(PREDICTIONS, CANDIDATES, OUTPUT, threshold=0.9)
    
    print(f"\n{'='*70}")
    print(f"✅ DONE")
    print(f"{'='*70}")
    print(f"Validated V-genes: {OUTPUT}")
    print(f"\nNext steps:")
    print(f"  1. Align with human V-genes")
    print(f"  2. Visual inspection in SeaView")
    print(f"  3. Annotation and classification")