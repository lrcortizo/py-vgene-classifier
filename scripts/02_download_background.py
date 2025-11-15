"""
Download background sequences from NCBI WGS and translate to amino acids
"""

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from pathlib import Path
import random
import time
import os
from dotenv import load_dotenv
import warnings
from Bio import BiopythonWarning

# Suppress partial codon warnings
warnings.simplefilter("ignore", BiopythonWarning)

# Load environment variables
load_dotenv()

NCBI_EMAIL = os.getenv("NCBI_EMAIL")
if not NCBI_EMAIL:
    raise ValueError("NCBI_EMAIL not found. Create .env file with your email.")

Entrez.email = NCBI_EMAIL

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "negative"
OUTPUT_FILE = OUTPUT_DIR / "background.fasta"
NUM_SEQUENCES = 42  # ration 3:1 to positive class (251 sequences)
TARGET_LENGTH_MIN = 103  # minimum length of positive class sequences
TARGET_LENGTH_MAX = 116  # maximum length of positive class sequences
CHROMOSOME = "NC_000001.11"  # Human chromosome 1 (largest)


def extract_peptides_from_protein(protein_seq, min_len, max_len):
    """Extract peptides of target length from a protein sequence"""
    peptides = []
    protein_str = str(protein_seq)

    # Split by stop codons
    fragments = protein_str.split("*")

    # For each fragment, extract windows of target size
    for fragment in fragments:
        if len(fragment) >= min_len:
            # Extract all possible windows
            for i in range(len(fragment) - min_len + 1):
                for target_len in range(
                    min_len, min(max_len + 1, len(fragment) - i + 1)
                ):
                    peptide = fragment[i : i + target_len]
                    if min_len <= len(peptide) <= max_len:
                        # Check if it doesn't have weird characters
                        if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in peptide):
                            peptides.append(peptide)

    return peptides


def fetch_random_genomic_region(chrom_id, dna_length=3600):
    """Fetch a random genomic region and translate it"""
    try:
        # Get chromosome length (cache it)
        if not hasattr(fetch_random_genomic_region, "chrom_length"):
            handle = Entrez.efetch(
                db="nucleotide", id=chrom_id, rettype="gb", retmode="xml"
            )
            record = Entrez.read(handle)
            handle.close()
            fetch_random_genomic_region.chrom_length = int(record[0]["GBSeq_length"])

        chrom_length = fetch_random_genomic_region.chrom_length

        # Fetch DNA region
        start = random.randint(1, chrom_length - dna_length)
        end = start + dna_length

        handle = Entrez.efetch(
            db="nucleotide",
            id=chrom_id,
            rettype="fasta",
            retmode="text",
            seq_start=start,
            seq_stop=end,
        )

        record = SeqIO.read(handle, "fasta")
        handle.close()

        # Translate in all 6 frames
        all_peptides = []

        # Forward strand
        for frame in range(3):
            seq_frame = record.seq[frame:]
            # Trim to multiple of 3
            seq_frame = seq_frame[: len(seq_frame) - len(seq_frame) % 3]
            if len(seq_frame) >= 3:
                protein = seq_frame.translate()
                all_peptides.extend(
                    extract_peptides_from_protein(
                        protein, TARGET_LENGTH_MIN, TARGET_LENGTH_MAX
                    )
                )

        # Reverse complement
        rev_seq = record.seq.reverse_complement()
        for frame in range(3):
            seq_frame = rev_seq[frame:]
            seq_frame = seq_frame[: len(seq_frame) - len(seq_frame) % 3]
            if len(seq_frame) >= 3:
                protein = seq_frame.translate()
                all_peptides.extend(
                    extract_peptides_from_protein(
                        protein, TARGET_LENGTH_MIN, TARGET_LENGTH_MAX
                    )
                )

        if all_peptides:
            # Pick random peptide
            peptide = random.choice(all_peptides)

            new_record = SeqIO.SeqRecord(
                Seq(peptide),
                id=f"bg_{len(peptide)}aa_{start}",
                description=f"Translated from {chrom_id}:{start}-{end}",
            )
            return new_record

        return None

    except Exception as e:
        print(f" ❌ Error: {str(e)[:50]}")
        return None


# Main
print("=" * 70)
print("DOWNLOADING & TRANSLATING BACKGROUND SEQUENCES FROM NCBI WGS")
print("=" * 70)
print(f"Target: {NUM_SEQUENCES} protein sequences")
print(f"Length range: {TARGET_LENGTH_MIN}-{TARGET_LENGTH_MAX} aa")
print(f"Source: Human chr1 ({CHROMOSOME})")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sequences = []
attempts = 0
max_attempts = NUM_SEQUENCES * 3  # Avoid infinite loops

while len(sequences) < NUM_SEQUENCES and attempts < max_attempts:
    attempts += 1

    print(f"[{len(sequences)+1}/{NUM_SEQUENCES}] Attempt {attempts}...", end="")

    seq = fetch_random_genomic_region(CHROMOSOME)

    if seq:
        sequences.append(seq)
        print(f" ✅ {len(seq.seq)} aa")
    else:
        print(" ⚠️")

    time.sleep(0.3)  # Faster

    if len(sequences) % 50 == 0 and len(sequences) > 0:
        SeqIO.write(sequences, OUTPUT_FILE, "fasta")
        print(f"   💾 Checkpoint: {len(sequences)} saved")

print("\n" + "=" * 70)
SeqIO.write(sequences, OUTPUT_FILE, "fasta")
print(f"✅ Saved {len(sequences)} sequences to {OUTPUT_FILE}")
print("=" * 70)
