import torch
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.features.encoding import sequences_to_tensor  # noqa: E402
from src.models.classifier import VGeneCNN  # noqa: E402

# Load data
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"

val_df = pd.read_csv(DATA_DIR / "val.csv")

# Load model
model = VGeneCNN(input_channels=20, seq_length=116, num_filters=[64, 128, 256])
model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt"))
model.eval()

# Encode
X_val = sequences_to_tensor(val_df["sequence"].tolist())

# Predict
with torch.no_grad():
    probs = model(X_val).numpy().flatten()

# Show some predictions
print("Sample predictions:")
print("=" * 70)

for i in range(min(10, len(val_df))):
    true_label = val_df.iloc[i]["label"]
    pred_prob = probs[i]
    pred_label = 1 if pred_prob > 0.5 else 0

    correct = "✅" if true_label == pred_label else "❌"

    print(
        f"{correct} True: {true_label} | Pred: {pred_label} (prob: {pred_prob:.4f}) | "
        f"Class: {val_df.iloc[i]['class']}"
    )

# Summary
print("\n" + "=" * 70)
print(f"Total val samples: {len(val_df)}")
print(f"Correct predictions: {sum((probs > 0.5).astype(int) == val_df['label'])}")
print(
    f"Accuracy: {sum((probs > 0.5).astype(int) == val_df['label']) / len(val_df):.4f}"
)
