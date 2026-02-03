#!/usr/bin/env python3
"""
Filter candidates using trained multiclass CNN.
Predicts locus for each V-gene candidate.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.encoding import sequences_to_tensor
from src.models.classifier import VGeneCNN

# Class names
CLASS_NAMES = ['background', 'IGHV', 'IGKV', 'TRAV', 'TRBV']

def load_model(model_path, device):
    """Load trained multiclass model."""
    model = VGeneCNN(
        input_channels=20,
        seq_length=116,
        num_filters=[64, 128, 256],
        kernel_size=3,
        dropout=0.3
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def predict_sequences(sequences, model, device, max_length=116, batch_size=64):
    """Predict class for each sequence."""
    results = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]

        # Encode
        seqs_str = [str(rec.seq) for rec in batch]
        tensors = sequences_to_tensor(seqs_str, max_length)
        tensors = tensors.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(tensors)  # (batch, 5) logits
            probs = torch.softmax(outputs, dim=1)  # Convert to probabilities

            # Get predicted class and probability
            max_probs, predictions = torch.max(probs, dim=1)

            probs_np = probs.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            max_probs_np = max_probs.cpu().numpy()

        # Store results
        for rec, pred_class, max_prob, all_probs in zip(batch, predictions_np, max_probs_np, probs_np):
            results.append({
                'record': rec,
                'predicted_class': int(pred_class),
                'predicted_locus': CLASS_NAMES[pred_class],
                'probability': float(max_prob),
                'prob_background': float(all_probs[0]),
                'prob_IGHV': float(all_probs[1]),
                'prob_IGKV': float(all_probs[2]),
                'prob_TRAV': float(all_probs[3]),
                'prob_TRBV': float(all_probs[4])
            })

    return results

def main():
    parser = argparse.ArgumentParser(description="Filter candidates with multiclass CNN")
    parser.add_argument("--candidates", required=True, help="Candidate sequences FASTA")
    parser.add_argument("--model", required=True, help="Trained multiclass model (.pt)")
    parser.add_argument("--output", required=True, help="Output FASTA for predicted V-genes")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum probability to classify as V-gene (non-background)")
    parser.add_argument("--save-all", action="store_true",
                        help="Save CSV with all predictions")
    parser.add_argument("--max-length", type=int, default=116, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model, device)
    print("Model loaded successfully")

    # Load candidates
    print(f"\nLoading candidates from {args.candidates}...")
    candidates = list(SeqIO.parse(args.candidates, "fasta"))
    print(f"Total candidates: {len(candidates)}")

    # Predict
    print("\nPredicting...")
    results = predict_sequences(candidates, model, device, args.max_length, args.batch_size)

    # Separate V-genes from background
    vgenes = []
    background = []

    for res in results:
        if res['predicted_class'] == 0:  # Background
            background.append(res)
        else:  # V-gene (any locus)
            if res['probability'] >= args.threshold:
                vgenes.append(res)
            else:
                background.append(res)

    # Statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total candidates: {len(results)}")
    print(f"\nPredicted V-genes: {len(vgenes)} ({len(vgenes)/len(results)*100:.1f}%)")
    print(f"Predicted background: {len(background)} ({len(background)/len(results)*100:.1f}%)")

    # V-genes by locus
    if vgenes:
        print("\nV-genes by locus:")
        locus_counts = {}
        for v in vgenes:
            locus = v['predicted_locus']
            locus_counts[locus] = locus_counts.get(locus, 0) + 1

        for locus in ['IGHV', 'IGKV', 'TRAV', 'TRBV']:
            count = locus_counts.get(locus, 0)
            print(f"  {locus}: {count}")

        # Probability distribution
        probs = [v['probability'] for v in vgenes]
        print(f"\nProbability distribution (V-genes):")
        print(f"  Min:    {min(probs):.4f}")
        print(f"  Max:    {max(probs):.4f}")
        print(f"  Mean:   {np.mean(probs):.4f}")
        print(f"  Median: {np.median(probs):.4f}")

        # Top predictions
        vgenes_sorted = sorted(vgenes, key=lambda x: x['probability'], reverse=True)
        print(f"\nTop 10 predictions:")
        for i, v in enumerate(vgenes_sorted[:10]):
            print(f"  {i+1}. {v['record'].id[:30]:30s} {v['predicted_locus']:5s} prob={v['probability']:.4f}")

    # Save V-genes FASTA
    if vgenes:
        vgene_records = []
        for v in vgenes:
            new_rec = SeqRecord(
                v['record'].seq,
                id=v['record'].id,
                description=f"{v['record'].description} predicted_locus={v['predicted_locus']} prob={v['probability']:.4f}"
            )
            vgene_records.append(new_rec)

        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        SeqIO.write(vgene_records, args.output, "fasta")
        print(f"\n✅ V-genes saved to: {args.output}")
    else:
        print("\n⚠️  No V-genes predicted above threshold")

    # Save all predictions to CSV
    if args.save_all:
        csv_output = args.output.replace('.fasta', '_all_predictions.csv')

        data = []
        for res in results:
            data.append({
                'id': res['record'].id,
                'sequence': str(res['record'].seq),
                'length': len(res['record'].seq),
                'predicted_class': res['predicted_class'],
                'predicted_locus': res['predicted_locus'],
                'probability': res['probability'],
                'prob_background': res['prob_background'],
                'prob_IGHV': res['prob_IGHV'],
                'prob_IGKV': res['prob_IGKV'],
                'prob_TRAV': res['prob_TRAV'],
                'prob_TRBV': res['prob_TRBV']
            })

        df = pd.DataFrame(data)
        df = df.sort_values('probability', ascending=False)
        df.to_csv(csv_output, index=False)
        print(f"✅ All predictions saved to: {csv_output}")

    print("\nDone!")

if __name__ == "__main__":
    main()
