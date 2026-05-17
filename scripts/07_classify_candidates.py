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
import re
import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.terminal_encoding import (TerminalRegionEncoder, TerminalRegionEncoderV3,
                                             MOTIF_FLAG_NAMES)
from src.models.cnn_terminal import CNN_TerminalEncoding

# Class names
CLASS_NAMES = ['background', 'IGHV', 'IGKV', 'TRAV', 'TRBV']

# FR1 start patterns (mirrored from 06_extract_candidates.py).
# Used to filter truncated/internal sequences before they reach the encoder.
FR1_PATTERNS = [
    # IGHV
    r'[QE]VQ[LV]', r'QVTL', r'QAYL', r'QREL',
    r'VQL[QV]',               # IGHV truncated: missing first aa (VQLQ/VQLV)
    r'SQL[QV]',               # IGHV truncated variant (SQLQ/SQLV)
    r'[QE]VQQ',               # IGHV variant (QVQQ/EVQQ)
    r'GAEL[VKR]',             # IGHV FR1 internal start (SGAELV/SGAELK/SGAELR)
    r'[EF][VF]KL[QE]Q',      # IGHV leader: EVKLQQ/EFKLQQ (IGHV1-39/43 family)
    # IGKV / IGLV
    r'D[ILV][VKQ][MLV]TQ', r'Q[AS]VL[TV]Q', r'Q[AS]V[LV]TQ', r'ETT[LV]TQ',
    r'[GQ][EI]NV[LI]TQ',     # IGKV4 family (GENVLTQ, QINVLTQ)
    # TRAV
    r'[AG][DNQ][SKTVRGN]V[TNVSA]Q', r'[GQKEI][QEKLMNIVDRSG][QKEVDLNIVGSA]V[EKD]Q',
    r'[GQK][QKE][QKVLN]V[QK]Q', r'[QE][QKE]L[NQKE][QS]',
    r'[IGSDV][DLSAV][AGS]K[TS][TQ]', r'[QS][QK][IK][KHE][QHF]',
    r'[QS]SP[QE]SLT',         # TRAV14 family (QSPQSLT, SSPESLT)
    # TRBV
    r'[DNEAGHSK][ADSGEP][GA][VIA][TISQAV]Q', r'D[ADGEPSVK][GAEVQKRDP][VI][TISQF]Q',
    r'DT[EGAKD][VI][TSFIQ]Q', r'[ENDHSAG][ASDHE][EAQKDGT][VI][TS]Q',
    r'GA[LMV][VLI][TSIQ]Q', r'[EA][PGS][EA][VI][TSIQ]Q', r'[AS]QT[ILV][HNQ]',
    r'D[VA][KRM][VI][TS]Q',
    # TRAV additional (v2.2.0)
    r'GQ[SNK][LIV][EQD]Q', r'KDQ[VI][FY]Q', r'[ED]NQ[VK][EQH]', r'R[KNQ]EV[EK]',
    r'GES[VT][GL]', r'AQK[IV][TI]Q', r'[SD]QQ[EGK][EQ]', r'KQE[VK][TQ]Q',
    r'GQQ[VK][MQVK]Q', r'DQQV[KR]Q', r'EDKV[VIMQ]', r'SNSV[KR]Q',
    # TRBV additional (v2.2.0)
    r'VTLLEQ', r'GPKVLQ', r'ETAVFQ', r'NTKITQ', r'[DN]SGVVQ', r'DTTVKQ',
    r'GGIITQ', r'GALVYQ', r'DAAVTQ', r'VAGVTQ', r'NSKVIQ', r'DMKVTQ', r'SVLLYQ',
]


def has_fr1_pattern(sequence: str, window: int = 30) -> bool:
    """Return True if any FR1 start pattern matches within the first `window` aa,
    or within positions 15-50 (handles ~15-25aa signal peptide prefixes)."""
    prefix = sequence[:window]
    if any(re.search(p, prefix) for p in FR1_PATTERNS):
        return True
    # Second pass: check window[15:50] to handle signal-peptide-prefixed candidates
    leader_window = sequence[15:50]
    return any(re.search(p, leader_window) for p in FR1_PATTERNS)


def calculate_confidence_level(prob, margin, motif_flags=None):
    if motif_flags is None:
        # fallback sin flags (encoder v2)
        if prob >= 0.99 and margin >= 0.95:
            return "high"
        elif prob >= 0.80 and margin >= 0.60:
            return "medium"
        else:
            return "low"

    n_motifs = sum(motif_flags.values())

    if prob >= 0.99 and n_motifs >= 4:
        return "high"
    elif prob >= 0.80 and n_motifs >= 2:
        return "medium"
    else:
        return "low"


def load_model(model_path, num_classes, device, input_size=2000):
    """Load trained model with terminal encoding."""
    print(f"   Loading model architecture...")
    model = CNN_TerminalEncoding(input_size=input_size, num_classes=num_classes)

    print(f"   Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model loaded: {n_params:,} parameters")

    return model


def predict_sequences(sequences, model, device, batch_size=64, encoder=None,
                      require_fr1=False, use_motif_flags=False):
    """Predict class for each sequence using terminal encoding."""
    if encoder is None:
        encoder = TerminalRegionEncoder()
    results = []
    fr1_filtered = 0

    print(f"\n   Processing {len(sequences):,} sequences in batches of {batch_size}...")
    if require_fr1:
        print(f"   FR1 filter: ON (window=30aa)")

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]

        # Encode with terminal-region method
        batch_encodings = []
        valid_records = []

        for rec in batch:
            # FR1 filter: sequences without a recognisable V-gene start in the
            # first 30aa are redirected to background without reaching the model.
            if require_fr1 and not has_fr1_pattern(str(rec.seq)):
                fr1_filtered += 1
                _fr1_flags = {n: False for n in MOTIF_FLAG_NAMES} if use_motif_flags else None
                results.append({
                    'record': rec,
                    'predicted_class': 0,
                    'predicted_locus': 'background',
                    'probability': 1.0,
                    'second_prob': 0.0,
                    'margin': 1.0,
                    'confidence_level': calculate_confidence_level(1.0, 1.0, _fr1_flags),
                    'motif_flags': _fr1_flags,
                    'prob_background': 1.0,
                    **{f'prob_{c}': 0.0 for c in CLASS_NAMES if c != 'background'},
                })
                continue
            try:
                if use_motif_flags:
                    encoding, flags = encoder.encode_with_flags(str(rec.seq))
                else:
                    encoding = encoder.encode(str(rec.seq))
                    flags = None
                batch_encodings.append(encoding)
                valid_records.append((rec, flags))
            except Exception as e:
                print(f"   Warning: Could not encode {rec.id}: {e}")
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

            # Second-highest probability for margin calculation
            sorted_probs_np = np.sort(probs_np, axis=1)[:, ::-1]
            second_probs_np = sorted_probs_np[:, 1]

        # Store results
        for (rec, flags), pred_class, max_prob, second_prob, all_probs in zip(
                valid_records, predictions_np, max_probs_np, second_probs_np, probs_np):
            margin = float(max_prob) - float(second_prob)
            result = {
                'record': rec,
                'predicted_class': int(pred_class),
                'predicted_locus': CLASS_NAMES[pred_class],
                'probability': float(max_prob),
                'second_prob': float(second_prob),
                'margin': margin,
                'confidence_level': calculate_confidence_level(float(max_prob), margin, flags),
                'motif_flags': flags,
            }

            # Add individual class probabilities
            for j, class_name in enumerate(CLASS_NAMES):
                result[f'prob_{class_name}'] = float(all_probs[j])

            results.append(result)

        # Progress
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed: {min(i+batch_size, len(sequences)):,}/{len(sequences):,}")

    if require_fr1 and fr1_filtered:
        print(f"   FR1 filter: {fr1_filtered:,} sequences redirected to background "
              f"(no FR1 pattern in first 30aa)")
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
    parser.add_argument("--encoder-version", choices=["v2", "v3"], default="v2",
                       help="Encoder version: v2=2000 dims (default), v3=2045 dims")
    parser.add_argument("--require-fr1", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Filter sequences without FR1 pattern in first 30aa. "
                             "Default: False. Use --require-fr1 to enable explicitly.")

    args = parser.parse_args()

    # Select encoder and input_size
    if args.encoder_version == "v3":
        encoder = TerminalRegionEncoderV3()
        input_size = 2045
    else:
        encoder = TerminalRegionEncoder()
        input_size = 2000

    require_fr1 = args.require_fr1

    print("=" * 80)
    print("V-GENE PREDICTION - v2.1.0")
    print("=" * 80)
    print(f"Encoding: Terminal-region {args.encoder_version.upper()} ({input_size} dims)")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"FR1 filter: {'ON' if require_fr1 else 'OFF'}")
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
    model = load_model(args.model, args.num_classes, device, input_size=input_size)
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
    use_motif_flags = (args.encoder_version == "v3")
    results = predict_sequences(candidates, model, device, args.batch_size,
                                encoder=encoder, require_fr1=require_fr1,
                                use_motif_flags=use_motif_flags)
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

        # Confidence level breakdown
        conf_counts = {'high': 0, 'medium': 0, 'low': 0}
        for v in vgenes:
            conf_counts[v['confidence_level']] += 1
        total_vg = len(vgenes)
        print("\n" + "-" * 80)
        print("CONFIDENCE LEVELS (V-genes only):")
        print("-" * 80)
        for level in ['high', 'medium', 'low']:
            cnt = conf_counts[level]
            pct = cnt / total_vg * 100 if total_vg > 0 else 0
            label = level.capitalize()
            print(f"  {label+':':<8} {cnt:4d} ({pct:5.1f}%)")

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
    csv_output = args.output.replace('.fasta', '_predictions.csv')
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
    if vgene_records or args.save_all:
        data = []
        for res in (vgenes + background if args.save_all else vgenes):
            row = {
                'id': res['record'].id,
                'sequence': str(res['record'].seq),
                'length': len(res['record'].seq),
                'predicted_class': res['predicted_class'],
                'predicted_locus': res['predicted_locus'],
                'probability': res['probability'],
                'second_prob': res['second_prob'],
                'margin': res['margin'],
                'confidence_level': res['confidence_level'],
            }
            # Add individual motif flags (v3 only; None → absent columns)
            if res['motif_flags'] is not None:
                for flag_name in MOTIF_FLAG_NAMES:
                    row[f'motif_{flag_name}'] = res['motif_flags'][flag_name]
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
