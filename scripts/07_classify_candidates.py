#!/usr/bin/env python3
"""
Script 07: Predict V-Gene Loci with Terminal-Region Encoding
Version: 2.1.0
Purpose: Classify candidate sequences using trained model with terminal encoding

Uses trained CNN model from script 03 to classify V-gene candidates extracted
from genome TBLASTN search. Outputs predicted V-genes by locus.

USAGE:
    python scripts/07_classify_candidates.py \
        --candidates results/mouse_v2/candidates.fasta \
        --model models/v2_hard_negatives/best_model.pt \
        --output results/mouse_v2/predicted_vgenes.fasta \
        --threshold 0.5 \
        --save-all
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
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.terminal_encoding import TerminalRegionEncoder
from src.models.cnn_terminal import CNN_TerminalEncoding

# Class names
CLASS_NAMES = ['background', 'IGHV', 'IGKV', 'TRAV', 'TRBV']


def load_model(model_path, num_classes, device):
    """Load trained model with terminal encoding."""
    print(f"   Loading model architecture...")
    model = CNN_TerminalEncoding(input_size=2000, num_classes=num_classes)

    print(f"   Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model loaded: {n_params:,} parameters")

    return model


def predict_sequences(sequences, model, device, batch_size=64):
    """Predict class for each sequence using terminal encoding."""
    encoder = TerminalRegionEncoder()
    results = []

    print(f"\n   Processing {len(sequences):,} sequences in batches of {batch_size}...")

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]

        # Encode with terminal-region method
        batch_encodings = []
        valid_records = []

        for rec in batch:
            try:
                encoding = encoder.encode(str(rec.seq))
                batch_encodings.append(encoding)
                valid_records.append(rec)
            except Exception as e:
                print(f"   ⚠️  Warning: Could not encode {rec.id}: {e}")
                continue

        if not batch_encodings:
            continue

        # Convert to tensor
        tensors = torch.FloatTensor(np.array(batch_encodings))
        tensors = tensors.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(tensors)  # (batch, num_classes) logits
            probs = torch.softmax(outputs, dim=1)  # Convert to probabilities

            # Get predicted class and probability
            max_probs, predictions = torch.max(probs, dim=1)

            probs_np = probs.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            max_probs_np = max_probs.cpu().numpy()

        # Store results
        for rec, pred_class, max_prob, all_probs in zip(valid_records, predictions_np, max_probs_np, probs_np):
            result = {
                'record': rec,
                'predicted_class': int(pred_class),
                'predicted_locus': CLASS_NAMES[pred_class],
                'probability': float(max_prob),
            }

            # Add individual class probabilities
            for j, class_name in enumerate(CLASS_NAMES):
                result[f'prob_{class_name}'] = float(all_probs[j])

            results.append(result)

        # Progress
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed: {min(i+batch_size, len(sequences)):,}/{len(sequences):,}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict V-gene loci with terminal-region encoding (v2.1.0)"
    )
    parser.add_argument("--candidates", required=True,
                       help="Candidate sequences FASTA")
    parser.add_argument("--model", required=True,
                       help="Trained model (.pt)")
    parser.add_argument("--output", required=True,
                       help="Output FASTA for predicted V-genes")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Minimum probability to classify as V-gene (default: 0.5)")
    parser.add_argument("--num-classes", type=int, default=5,
                       help="Number of classes (default: 5)")
    parser.add_argument("--save-all", action="store_true",
                       help="Save CSV with all predictions")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size (default: 64)")

    args = parser.parse_args()

    print("=" * 80)
    print("V-GENE PREDICTION - v2.1.0")
    print("=" * 80)
    print(f"Encoding: Terminal-region (N/C-term + dipeptides)")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)
    print()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    print("🔨 LOADING MODEL")
    print("-" * 80)
    model = load_model(args.model, args.num_classes, device)
    print()

    # Load candidates
    print("📖 LOADING CANDIDATES")
    print("-" * 80)
    print(f"   Reading from: {args.candidates}")
    candidates = list(SeqIO.parse(args.candidates, "fasta"))
    print(f"   Total candidates: {len(candidates):,}")
    print()

    # Predict
    print("🔮 PREDICTING")
    print("-" * 80)
    results = predict_sequences(candidates, model, device, args.batch_size)
    print(f"   ✅ Predictions complete: {len(results):,} sequences")
    print()

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
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total predictions: {len(results):,}")
    print(f"\nPredicted V-genes: {len(vgenes):,} ({len(vgenes)/len(results)*100:.1f}%)")
    print(f"Predicted background: {len(background):,} ({len(background)/len(results)*100:.1f}%)")

    # V-genes by locus
    if vgenes:
        print("\n" + "-" * 80)
        print("V-GENES BY LOCUS")
        print("-" * 80)
        locus_counts = {}
        for v in vgenes:
            locus = v['predicted_locus']
            locus_counts[locus] = locus_counts.get(locus, 0) + 1

        for locus in ['IGHV', 'IGKV', 'TRAV', 'TRBV']:
            count = locus_counts.get(locus, 0)
            pct = count / len(vgenes) * 100 if len(vgenes) > 0 else 0
            print(f"  {locus}: {count:4d} ({pct:5.1f}%)")

        # Probability distribution
        probs = [v['probability'] for v in vgenes]
        print("\n" + "-" * 80)
        print("PROBABILITY DISTRIBUTION (V-genes)")
        print("-" * 80)
        print(f"  Min:    {min(probs):.4f}")
        print(f"  Max:    {max(probs):.4f}")
        print(f"  Mean:   {np.mean(probs):.4f}")
        print(f"  Median: {np.median(probs):.4f}")

        # Top predictions
        vgenes_sorted = sorted(vgenes, key=lambda x: x['probability'], reverse=True)
        print("\n" + "-" * 80)
        print("TOP 10 PREDICTIONS")
        print("-" * 80)
        for i, v in enumerate(vgenes_sorted[:10]):
            print(f"  {i+1:2d}. {v['record'].id[:40]:40s} {v['predicted_locus']:5s} prob={v['probability']:.4f}")
    else:
        print("\n⚠️  No V-genes predicted above threshold")

    # Save V-genes FASTA
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    vgene_records = []
    if vgenes:
        for v in vgenes:
            new_rec = SeqRecord(
                v['record'].seq,
                id=v['record'].id,
                description=f"{v['record'].description} predicted_locus={v['predicted_locus']} prob={v['probability']:.4f}"
            )
            vgene_records.append(new_rec)

        # Create output directory
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

        SeqIO.write(vgene_records, args.output, "fasta")
        print(f"✅ V-genes FASTA: {args.output}")
    else:
        print("⚠️  No V-genes to save")

    # Save all predictions to CSV
    if vgene_records:  # Always save CSV
        csv_output = args.output.replace('.fasta', '_predictions.csv')

        data = []
        for res in (vgenes + background if args.save_all else vgenes):
            row = {
                'id': res['record'].id,
                'sequence': str(res['record'].seq),
                'length': len(res['record'].seq),
                'predicted_class': res['predicted_class'],
                'predicted_locus': res['predicted_locus'],
                'probability': res['probability'],
            }
            # Add class probabilities
            for class_name in CLASS_NAMES:
                row[f'prob_{class_name}'] = res[f'prob_{class_name}']

            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('probability', ascending=False)
        df.to_csv(csv_output, index=False)
        print(f"✅ All predictions CSV: {csv_output}")

    print()
    print("=" * 80)
    print("✅ PREDICTION COMPLETE")
    print("=" * 80)
    print()
    print("Next step: Validate predictions")
    print(f"  python scripts/08_validate_predictions.py \\")
    print(f"      --predictions {args.output} \\")
    print(f"      --predictions-csv {csv_output} \\")
    print(f"      --reference data/reference/<imgt_species>/all_vgenes_imgt.fasta \\")
    print(f"      --output-dir {os.path.dirname(args.output)}/validation")
    print()


if __name__ == "__main__":
    main()
