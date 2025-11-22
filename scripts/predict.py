"""
Classify new protein sequences as V-gene or background
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.encoding import sequences_to_tensor
from src.models.classifier import VGeneCNN

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load trained model"""
    # Create the architecture (must be EXACTLY the same as in training)
    model = VGeneCNN(
        input_channels=20,
        seq_length=116,
        num_filters=[64, 128, 256],
        kernel_size=3,
        dropout=0.3
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Set to evaluation mode (disables dropout)
    model.eval()
    
    # Move to device
    model.to(DEVICE)
    
    return model

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sequences(sequences, model):
    """
    Predict if sequences are V-genes or background
    
    Args:
        sequences: List of amino acid sequences (strings)
        model: Trained VGeneCNN model
    
    Returns:
        List of tuples: (sequence, probability, prediction)
    """
    # Encode sequences
    X = sequences_to_tensor(sequences)
    X = X.to(DEVICE)
    
    # Make predictions (without computing gradients)
    with torch.no_grad():
        probabilities = model(X).cpu().numpy().flatten()
    
    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).astype(int)
    
    # Prepare results
    results = []
    for seq, prob, pred in zip(sequences, probabilities, predictions):
        label = "V-gene" if pred == 1 else "background"
        results.append({
            'sequence': seq,
            'probability': prob,
            'prediction': pred,
            'label': label
        })
    
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("V-REGION CLASSIFIER - PREDICTION")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model()
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Device: {DEVICE}")
    
    # ========================================================================
    # EXAMPLE: Predict some test sequences
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    # Example sequences (you can change them)
    test_sequences = [
        # Typical V-gene (should predict ~1.0)
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGRINPN",
        
        # Another V-gene (should predict ~1.0)
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSG",
        
        # Random background sequence (should predict ~0.0)
        "MKTAEDGVLISRPDGLKPVQALMDEGTFVCRETSYRGAYHQDSPQAQYVLNEIQ",
        
        # Another background sequence (should predict ~0.0)
        "ASIGRPDGLCKPVQALMDEGTFVCRETSYRGAYHQDSPQAQYVLNEIQRSTWLP"
    ]
    
    # Make predictions
    results = predict_sequences(test_sequences, model)
    
    # Show results
    for i, result in enumerate(results, 1):
        seq_short = result['sequence'][:50] + "..." if len(result['sequence']) > 50 else result['sequence']
        print(f"\nSequence {i}:")
        print(f"  {seq_short}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Prediction: {result['label']}")
        
        # Visual confidence indicator
        if result['probability'] > 0.9:
            confidence = "🟢 Very high confidence"
        elif result['probability'] > 0.7:
            confidence = "🟡 High confidence"
        elif result['probability'] > 0.5:
            confidence = "🟠 Medium confidence"
        elif result['probability'] > 0.3:
            confidence = "🟠 Medium confidence"
        elif result['probability'] > 0.1:
            confidence = "🟡 High confidence (negative)"
        else:
            confidence = "🟢 Very high confidence (negative)"
        
        print(f"  Confidence: {confidence}")
    
    print("\n" + "=" * 70)
    print("✅ Predictions complete!")
    print("=" * 70)