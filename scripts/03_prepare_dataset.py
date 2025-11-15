"""
Prepare dataset: combine positive + negative, create train/val split
"""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import random

# Set seed for reproducibility
random.seed(42)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "raw" / "positive"
NEGATIVE_FILE = DATA_DIR / "raw" / "negative" / "background.fasta"
OUTPUT_DIR = DATA_DIR / "processed"

# Positive files
POSITIVE_FILES = [
    "ighv.fasta",
    "igkv.fasta",
    "iglv.fasta",
    "trav.fasta",
    "trbv.fasta",
    "trdv.fasta",
    "trgv.fasta",
]


def load_sequences(fasta_files, label, label_name):
    """Load sequences from FASTA files and assign labels"""
    data = []

    for fasta_file in fasta_files:
        if isinstance(fasta_file, Path):
            path = fasta_file
        else:
            path = POSITIVE_DIR / fasta_file

        if not path.exists():
            print(f"⚠️  {path.name} not found, skipping...")
            continue

        for record in SeqIO.parse(path, "fasta"):
            data.append(
                {
                    "id": record.id,
                    "sequence": str(record.seq),
                    "length": len(record.seq),
                    "label": label,
                    "class": label_name,
                    "source": path.name,
                }
            )

    return data


print("=" * 70)
print("PREPARING DATASET")
print("=" * 70)

# Load positive sequences (V genes)
print("\n📁 Loading positive sequences (V genes)...")
positive_data = load_sequences(POSITIVE_FILES, label=1, label_name="V-gene")
print(f"   Loaded {len(positive_data)} positive sequences")

# Load negative sequences (background)
print("\n📁 Loading negative sequences (background)...")
negative_data = load_sequences([NEGATIVE_FILE], label=0, label_name="background")
print(f"   Loaded {len(negative_data)} negative sequences")

# Combine
all_data = positive_data + negative_data
df = pd.DataFrame(all_data)

print(f"\n📊 Total dataset: {len(df)} sequences")
pos_pct = len(positive_data) / len(df) * 100
print(f"   Positive (V-gene): {len(positive_data)} ({pos_pct:.1f}%)")
neg_pct = len(negative_data) / len(df) * 100
print(f"   Negative (background): {len(negative_data)} ({neg_pct:.1f}%)")

# Train/val split (80/20) stratified by label
print("\n✂️  Splitting into train (80%) and validation (20%)...")
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

print(f"\n📊 Training set: {len(train_df)} sequences")
pos_train = sum(train_df["label"] == 1)
pct_pos_train = (pos_train / len(train_df)) * 100
print(f"   Positive: {pos_train} ({pct_pos_train:.1f}%)")
neg_train = sum(train_df["label"] == 0)
pct_neg_train = (neg_train / len(train_df)) * 100
print(f"   Negative: {neg_train} ({pct_neg_train:.1f}%)")

print(f"\n📊 Validation set: {len(val_df)} sequences")
pos_val = sum(val_df["label"] == 1)
pct_pos_val = (pos_val / len(val_df)) * 100
print(f"   Positive: {pos_val} ({pct_pos_val:.1f}%)")
neg_val = sum(val_df["label"] == 0)
pct_neg_val = (neg_val / len(val_df)) * 100
print(f"   Negative: {neg_val} ({pct_neg_val:.1f}%)")

# Save to CSV
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_csv = OUTPUT_DIR / "train.csv"
val_csv = OUTPUT_DIR / "val.csv"
full_csv = OUTPUT_DIR / "full_dataset.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
df.to_csv(full_csv, index=False)

print("\n💾 Saved CSV files:")
print(f"   {train_csv}")
print(f"   {val_csv}")
print(f"   {full_csv}")

# Also save as FASTA for convenience
train_fasta = OUTPUT_DIR / "train.fasta"
val_fasta = OUTPUT_DIR / "val.fasta"


def save_fasta(dataframe, output_path):
    """Save dataframe to FASTA format"""
    records = []
    for _, row in dataframe.iterrows():
        record = SeqRecord(
            Seq(row["sequence"]),
            id=f"{row['id']}|label={row['label']}",
            description=f"{row['class']} from {row['source']}",
        )
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


save_fasta(train_df, train_fasta)
save_fasta(val_df, val_fasta)

print("\n💾 Saved FASTA files:")
print(f"   {train_fasta}")
print(f"   {val_fasta}")

print("\n" + "=" * 70)
print("✅ Dataset preparation complete!")
print("=" * 70)

# Summary statistics
print("\n📈 Length statistics by class:")
print(df.groupby("class")["length"].describe())
