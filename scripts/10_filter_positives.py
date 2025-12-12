#!/usr/bin/env python3
"""
Filter candidate sequences using trained CNN model.

This script uses the trained multi-species CNN to classify
candidate sequences as V-genes or non-V-genes.

Usage:
    python scripts/10_filter_positives.py \
        --candidates results/bat/candidates.fasta \
        --model models/best_model_multispecies.pt \
        --output results/bat/vgenes_predicted.fasta \
        --threshold 0.5
"""

import argparse
import sys
from pathlib import Path
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.encoding import sequences_to_tensor
from src.models.classifier import VGeneCNN


def load_model(model_path: Path, device: torch.device):
    """Load trained CNN model."""
    print(f"\n🏗️  Loading model...")
    print(f"   Path: {model_path}")

    # Initialize model
    model = VGeneCNN(
        input_channels=20,
        seq_length=116,
        num_filters=[64, 128, 256],
        kernel_size=3,
        dropout=0.3
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"   ✅ Model loaded")

    return model


def predict_sequences(
    sequences: list,
    model,
    device: torch.device,
    max_length: int = 116,
    batch_size: int = 64
):
    """
    Predict probabilities for sequences.

    Returns list of (sequence, probability) tuples.
    """
    print(f"\n🔮 Predicting with CNN...")
    print(f"   Sequences: {len(sequences)}")
    print(f"   Batch size: {batch_size}")

    results = []

    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]

        # Encode
        tensors = sequences_to_tensor([str(rec.seq) for rec in batch], max_length=max_length)
        tensors = tensors.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(tensors).squeeze()

            # Handle single sequence
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            probs = outputs.cpu().numpy()

        # Store results
        for rec, prob in zip(batch, probs):
            results.append((rec, float(prob)))

        if (i + batch_size) % 500 == 0:
            print(f"   Processed {min(i+batch_size, len(sequences))}/{len(sequences)}...")

    print(f"   ✅ Predictions complete")

    return results


def filter_by_threshold(
    results: list,
    threshold: float = 0.5
):
    """Filter predictions by threshold."""
    print(f"\n📊 Filtering predictions...")
    print(f"   Threshold: {threshold}")

    positives = [(rec, prob) for rec, prob in results if prob >= threshold]
    negatives = [(rec, prob) for rec, prob in results if prob < threshold]

    print(f"   ✅ V-genes (≥{threshold}): {len(positives)}")
    print(f"   ❌ Non-V-genes (<{threshold}): {len(negatives)}")

    return positives, negatives


def main():
    parser = argparse.ArgumentParser(
        description="Filter candidates using trained CNN"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Candidate sequences FASTA file"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Trained model file (.pt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output FASTA file for predicted V-genes"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=116,
        help="Maximum sequence length (default: 116)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction (default: 64)"
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all predictions with probabilities in CSV"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FILTER V-GENE CANDIDATES WITH CNN")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # Load candidates
    print(f"\n📖 Loading candidates...")
    print(f"   File: {args.candidates}")

    candidates = list(SeqIO.parse(args.candidates, "fasta"))
    print(f"   ✅ Loaded {len(candidates)} candidates")

    # Length stats
    lengths = [len(rec.seq) for rec in candidates]
    print(f"   Length range: {min(lengths)}-{max(lengths)} aa")
    print(f"   Mean length: {sum(lengths)/len(lengths):.1f} aa")

    # Load model
    model = load_model(args.model, device)

    # Predict
    results = predict_sequences(
        candidates,
        model,
        device,
        args.max_length,
        args.batch_size
    )

    # Filter
    positives, negatives = filter_by_threshold(results, args.threshold)

    # Sort positives by probability (highest first)
    positives.sort(key=lambda x: x[1], reverse=True)

    # Save positives
    args.output.parent.mkdir(parents=True, exist_ok=True)

    positive_records = []
    for rec, prob in positives:
        new_rec = SeqRecord(
            rec.seq,
            id=rec.id,
            description=f"{rec.description} prob={prob:.4f}"
        )
        positive_records.append(new_rec)

    SeqIO.write(positive_records, args.output, "fasta")

    # Save all predictions to CSV if requested
    if args.save_all:
        csv_file = args.output.parent / f"{args.output.stem}_all_predictions.csv"

        data = []
        for rec, prob in results:
            data.append({
                "id": rec.id,
                "description": rec.description,
                "sequence": str(rec.seq),
                "length": len(rec.seq),
                "probability": prob,
                "predicted_class": "V-gene" if prob >= args.threshold else "non-V-gene"
            })

        df = pd.DataFrame(data)
        df = df.sort_values("probability", ascending=False)
        df.to_csv(csv_file, index=False)

        print(f"\n💾 All predictions saved: {csv_file}")

    # Summary statistics
    probs = [prob for _, prob in results]

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total candidates: {len(candidates)}")
    print(f"   Predicted V-genes: {len(positives)} ({len(positives)/len(candidates)*100:.1f}%)")
    print(f"   Predicted non-V-genes: {len(negatives)} ({len(negatives)/len(candidates)*100:.1f}%)")
    print(f"\n   Probability distribution:")
    print(f"   Min: {min(probs):.4f}")
    print(f"   Max: {max(probs):.4f}")
    print(f"   Mean: {sum(probs)/len(probs):.4f}")
    print(f"   Median: {sorted(probs)[len(probs)//2]:.4f}")

    # Top predictions
    print(f"\n   Top 10 V-gene predictions:")
    for i, (rec, prob) in enumerate(positives[:10], 1):
        print(f"   {i:2d}. {rec.id:20s} {prob:.4f} ({len(rec.seq)} aa)")

    print(f"\n💾 Predicted V-genes saved: {args.output}")

    # Comparison with previous results
    print(f"\n📈 Comparison:")
    print(f"   Previous attempt (human-only model): 11 genes")
    print(f"   Current attempt (multi-species model): {len(positives)} genes")
    print(f"   Improvement: {len(positives)/11:.1f}x more genes found!")


if __name__ == "__main__":
    main()
