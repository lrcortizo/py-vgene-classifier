"""
Classify sequences from a FASTA file
"""
import torch
from Bio import SeqIO
import pandas as pd
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from src.features.encoding import sequences_to_tensor
from src.models.classifier import VGeneCNN

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load trained model"""
    model = VGeneCNN(
        input_channels=20,
        seq_length=116,
        num_filters=[64, 128, 256],
        kernel_size=3,
        dropout=0.3
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

def predict_fasta(fasta_path, model, output_csv=None, threshold=0.5):
    """
    Classify all sequences in a FASTA file
    
    Args:
        fasta_path: Path to FASTA file
        model: Trained model
        output_csv: Optional path to save results as CSV
        threshold: Classification threshold
    
    Returns:
        DataFrame with predictions
    """
    print(f"\n📄 Reading FASTA file: {fasta_path}")
    
    # Read sequences from FASTA
    sequences = []
    seq_ids = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)
    
    print(f"   Found {len(sequences)} sequences")
    
    # Make predictions in batches (for efficiency)
    print("\n🔮 Making predictions...")
    
    batch_size = 32
    all_probs = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        X = sequences_to_tensor(batch_seqs).to(DEVICE)
        
        with torch.no_grad():
            probs = model(X).cpu().numpy().flatten()
        
        all_probs.extend(probs)
        
        # Progress
        progress = min(i + batch_size, len(sequences))
        print(f"   Progress: {progress}/{len(sequences)}", end='\r')
    
    print(f"\n   ✅ All predictions complete")
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'sequence_id': seq_ids,
        'sequence': sequences,
        'length': [len(s) for s in sequences],
        'probability': all_probs,
        'prediction': ['V-gene' if p > threshold else 'background' for p in all_probs]
    })
    
    # Statistics
    n_vgenes = (df['probability'] > threshold).sum()
    n_background = (df['probability'] <= threshold).sum()
    
    print(f"\n📊 Results:")
    print(f"   V-genes predicted: {n_vgenes} ({n_vgenes/len(df)*100:.1f}%)")
    print(f"   Background predicted: {n_background} ({n_background/len(df)*100:.1f}%)")
    print(f"   Mean probability (V-genes): {df[df['probability'] > threshold]['probability'].mean():.4f}")
    print(f"   Mean probability (background): {df[df['probability'] <= threshold]['probability'].mean():.4f}")
    
    # Save CSV if specified
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to: {output_csv}")
    
    return df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify sequences from FASTA')
    parser.add_argument('input_fasta', type=str, help='Input FASTA file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file (optional)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Probability threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("V-REGION CLASSIFIER - FASTA PREDICTION")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model()
    print(f"✅ Model loaded successfully")
    
    # Classify sequences
    df = predict_fasta(args.input_fasta, model, args.output, args.threshold)
    
    # Show some examples
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (first 10)")
    print("=" * 70)
    
    for idx, row in df.head(10).iterrows():
        seq_short = row['sequence'][:40] + "..." if len(row['sequence']) > 40 else row['sequence']
        print(f"\n{row['sequence_id']}")
        print(f"  Sequence: {seq_short}")
        print(f"  Probability: {row['probability']:.4f}")
        print(f"  Prediction: {row['prediction']}")
    
    print("\n" + "=" * 70)
    print("✅ Done!")
    print("=" * 70)