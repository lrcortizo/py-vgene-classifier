# V-Gene Classifier

Binary classifier to identify adaptive immune system V genes using PyTorch.

## Objective
Train a model to distinguish V genes (IGHV, IGKV, IGLV, TRAV, TRBV, TRDV, TRGV) from genomic background sequences.

## Dataset
- **Positive class**: Human V gene segments from adaptive immune receptor loci
- **Negative class**: Random genomic sequences from NCBI (3:1 ratio)
- **Split**: 80% training, 20% validation

## Setup
```bash