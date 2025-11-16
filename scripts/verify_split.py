import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

# Verify duplicate sequences between train and val sets
train_seqs = set(train_df["sequence"])
val_seqs = set(val_df["sequence"])

overlap = train_seqs.intersection(val_seqs)

print(f"Train sequences: {len(train_seqs)}")
print(f"Val sequences: {len(val_seqs)}")
print(f"Overlapping sequences: {len(overlap)}")

if len(overlap) > 0:
    print("\n⚠️  WARNING: There are overlapping sequences!")
    print("This could inflate validation performance.")
else:
    print("\n✅ No overlap - split is clean!")

# Verify length distribution
print("\nLength statistics:")
print("Train V-genes:", train_df[train_df["label"] == 1]["length"].describe())
print("Train background:", train_df[train_df["label"] == 0]["length"].describe())
print("Val V-genes:", val_df[val_df["label"] == 1]["length"].describe())
print("Val background:", val_df[val_df["label"] == 0]["length"].describe())
