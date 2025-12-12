# V-Gene Classifier - Multi-Species CNN

Deep learning pipeline for V-gene discovery in vertebrate genomes using multi-species trained CNN.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to identify V gene segments (variable genes from immunoglobulin and T-cell receptor loci) in vertebrate genomes. The classifier uses one-hot encoded amino acid sequences and achieves high performance through pattern recognition of characteristic V-gene motifs.

**NEW in v1.1.0**: Multi-species training pipeline with complete genome-to-prediction workflow, achieving significant improvement in V-gene discovery compared to single-species approaches.

## Key Features

- **Multi-Species Training**: Supports training on V-genes from multiple vertebrate species
- **Complete Pipeline**: From genome download to final V-gene predictions
- **GPU Acceleration**: CUDA support for faster training and inference
- **Synthetic Background**: Automated generation of negative training examples
- **TBLASTN Integration**: Candidate identification from whole genomes
- **High Accuracy**: Achieves 100% validation accuracy on multi-species datasets
- **Modular Design**: Reusable components for encoding, model architecture, and training
- **Reproducible Results**: Fixed random seeds and documented parameters

## What's New in v1.1.0

### Major Improvements
- **Multi-species dataset support**: Train on V-genes from diverse vertebrate species
- **Complete discovery pipeline**: End-to-end workflow from genome to predictions
- **Synthetic background generation**: Fast, reproducible negative training examples
- **Significant improvement**: Multi-species model finds substantially more V-genes than single-species models

### New Scripts
- `05_prepare_multispecies_dataset.py` - Multi-species dataset preparation
- `05b_generate_synthetic_background.py` - Synthetic background generation
- `06_train_multispecies_cnn.py` - Train on multi-species data
- `07_download_genome.py` - Automated genome download
- `08_run_tblastn.py` - TBLASTN search for candidates
- `09_extract_candidates.py` - Extract and translate sequences
- `10_filter_positives.py` - CNN-based filtering

## Project Structure
```
vgene-classifier/
│
├── data/
│   ├── raw/
│   │   ├── positive/          # V gene FASTA files or queries
│   │   └── negative/          # Background sequences
│   ├── processed/             # Train/val splits (CSV + FASTA)
│   └── genomes/               # Downloaded genome assemblies
│
├── src/
│   ├── features/
│   │   └── encoding.py        # One-hot encoding functions
│   └── models/
│       ├── classifier.py      # CNN architecture
│       └── train.py           # Training/evaluation functions
│
├── scripts/
│   ├── 01_explore_vgenes.py       # Data exploration
│   ├── 02_download_background.py  # NCBI background download
│   ├── 03_prepare_dataset.py      # Train/val split (single-species)
│   ├── 04_train_classifier.py     # Model training (single-species)
│   ├── 05_prepare_multispecies_dataset.py  # Multi-species dataset
│   ├── 05b_generate_synthetic_background.py # Synthetic negatives
│   ├── 06_train_multispecies_cnn.py        # Multi-species training
│   ├── 07_download_genome.py               # Genome download
│   ├── 08_run_tblastn.py                   # TBLASTN search
│   ├── 09_extract_candidates.py            # Candidate extraction
│   ├── 10_filter_positives.py              # CNN filtering
│   ├── verify_split.py            # Data quality checks
│   └── inspect_predictions.py     # Prediction analysis
│
├── models/
│   ├── best_model.pt              # Single-species trained model
│   └── best_model_multispecies.pt # Multi-species trained model
│
├── results/
│   ├── training_history.png       # Training curves
│   └── training_history_multispecies.png
│
├── .gitignore
├── requirements.txt
├── environment.yml
├── CHANGELOG.md
└── README.md
```

## Requirements

### Software
- Python 3.11+
- BLAST+ (2.10.0+)
- conda/mamba (recommended)

### Python Packages
```
biopython
pandas
numpy
scikit-learn
torch (with CUDA support recommended)
matplotlib
```

## Installation

### Option 1: Using Conda (Recommended)
```bash
# Create environment from file
conda env create -f environment.yml
conda activate vgene

# Or create manually
conda create -n vgene python=3.11 -y
conda activate vgene

# Install BLAST+
conda install -c bioconda blast

# Install PyTorch with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install biopython pandas numpy matplotlib scikit-learn
```

### Option 2: Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install BLAST+ separately
# Ubuntu/Debian: sudo apt install ncbi-blast+
# macOS: brew install blast
# Windows: Download from NCBI
```

### Environment Variables

Create a `.env` file (optional, only for NCBI downloads):
```
NCBI_EMAIL=your.email@example.com
```

## Quick Start - Multi-Species Pipeline

### Complete V-Gene Discovery Workflow
```bash
# 1. Generate synthetic background (~10 seconds)
python scripts/05b_generate_synthetic_background.py

# 2. Prepare multi-species dataset (~2 minutes)
python scripts/05_prepare_multispecies_dataset.py \
    --input-dir /path/to/annotated_vgenes \
    --loci ighv igkv trav trbv \
    --output-dir data/processed

# 3. Train CNN on multi-species data (~1 hour with GPU)
python scripts/06_train_multispecies_cnn.py \
    --train-csv data/processed/train_multispecies.csv \
    --val-csv data/processed/val_multispecies.csv \
    --output-dir models \
    --batch-size 64 \
    --epochs 50

# 4. Download target genome (~5-10 minutes)
python scripts/07_download_genome.py \
    --accession GCF_XXXXXXXXX.X \
    --output-dir data/genomes/target_species

# 5. Run TBLASTN search (~5-15 minutes)
python scripts/08_run_tblastn.py \
    --genome data/genomes/target_species/genome.fna \
    --query data/raw/positive/vgene_queries.fasta \
    --output results/target/tblastn_results.txt \
    --evalue 1e-5 \
    --threads 8

# 6. Extract candidate sequences (~5-10 minutes)
python scripts/09_extract_candidates.py \
    --tblastn-results results/target/tblastn_results.txt \
    --genome data/genomes/target_species/genome.fna \
    --output results/target/candidates.fasta

# 7. Filter with CNN (~1-2 minutes)
python scripts/10_filter_positives.py \
    --candidates results/target/candidates.fasta \
    --model models/best_model_multispecies.pt \
    --output results/target/vgenes_predicted.fasta \
    --threshold 0.5 \
    --save-all
```

**Total time**: ~2-3 hours (mostly automated)

## Usage - Original Pipeline (Single-Species)

### 1. Explore V Gene Data
```bash
python scripts/01_explore_vgenes.py
```

### 2. Download Background Sequences
```bash
python scripts/02_download_background.py
```

### 3. Prepare Dataset
```bash
python scripts/03_prepare_dataset.py
```

### 4. Train Model
```bash
python scripts/04_train_classifier.py
```

## Data Requirements

### Multi-Species Pipeline

**Input**: Annotated V-gene FASTAs from multiple species
- Format: Standard FASTA with protein sequences
- Source: IMGT alignment, custom annotation pipeline, or literature
- Recommended loci: IGHV, IGKV, IGLV, TRAV, TRBV

**Queries**: Representative V-genes for TBLASTN
- Typically 100-200 sequences per target genome
- Can be generated from training dataset
- Amino acid sequences

### Original Pipeline

**Positive Class**: V gene FASTA files in `data/raw/positive/`
```
>V-gene_1
QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMH...
>V-gene_2
DIQMTQSPSSLSASVGDRVTITCRASQSISSWLA...
```

**Negative Class**: Background sequences (auto-generated or provided)

## Model Architecture

### CNN Design
```
Input (batch, 20, max_length)
  ↓
Conv1D(20→64, k=3) + BatchNorm + ReLU + MaxPool(2)
  ↓
Conv1D(64→128, k=3) + BatchNorm + ReLU + MaxPool(2)
  ↓
Conv1D(128→256, k=3) + BatchNorm + ReLU + MaxPool(2)
  ↓
Flatten
  ↓
FC(flatten_size→128) + ReLU + Dropout(0.3)
  ↓
FC(128→64) + ReLU + Dropout(0.3)
  ↓
FC(64→1) + Sigmoid
  ↓
Output: V-gene probability (0-1)
```

**Parameters**: 595,265 trainable
**Input**: One-hot encoded protein sequences (20 amino acids × sequence length)

### Performance

Typical performance on multi-species validation:
- Training accuracy: ~99-100%
- Validation accuracy: ~97-100%
- F1 Score: ~0.97-1.0
- AUC-ROC: ~0.99-1.0
- Convergence: 10-25 epochs

## Methodology

### Feature Representation
- **Encoding**: One-hot (20-dimensional per amino acid)
- **Max length**: 116 amino acids (configurable)
- **Padding**: Zero-padding for shorter sequences

### Training Configuration
```python
BATCH_SIZE = 64         # 32 for single-species
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
OPTIMIZER = Adam
LOSS = Binary Cross Entropy
```

### Evaluation Metrics
- Accuracy, Precision, Recall
- F1 Score (primary metric)
- AUC-ROC

## Customization

### Adjust Model Architecture
Edit `src/models/classifier.py`:
```python
model = VGeneCNN(
    input_channels=20,          # Number of amino acids
    seq_length=116,             # Maximum sequence length
    num_filters=[64, 128, 256], # Filters per conv layer
    kernel_size=3,              # Kernel size
    dropout=0.3                 # Dropout rate
)
```

### Change CNN Threshold
```bash
python scripts/10_filter_positives.py \
    --threshold 0.3  # Lower = more sensitive, more false positives
```

### Modify TBLASTN Sensitivity
```bash
python scripts/08_run_tblastn.py \
    --evalue 1e-3  # More permissive than default 1e-5
```

## Biological Interpretation

### Why V Genes Are Distinguishable

V genes typically contain:
1. **Framework regions (FR)**: Conserved amino acid patterns
2. **CDR loops**: Variable but structurally constrained regions
3. **Recombination signal sequences**: Characteristic motifs
4. **Immunoglobulin/TCR domains**: Specific protein architecture

The CNN learns to detect these patterns through convolutional filters.

### Multi-Species Advantages

Training on multiple species improves:
- **Generalization**: Learns universal V-gene features
- **Sensitivity**: Adapts to lineage-specific variations
- **Discovery**: Better performance on non-model organisms

## Troubleshooting

### BLAST Not Found
```bash
# Install BLAST+
conda install -c bioconda blast
# Or: sudo apt install ncbi-blast+
```

### GPU Not Detected
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory During Training
```python
# Reduce batch size in training scripts
BATCH_SIZE = 32  # or 16
```

### Low Performance on Target Species
- Include more diverse species in training
- Lower CNN threshold (try 0.3 or 0.4)
- Use species-specific queries if available
- Check genome assembly quality

### TBLASTN Finds No Hits
- Verify query sequences are in amino acid format
- Try more permissive e-value (1e-3)
- Check genome FASTA format

## Citation

If you use this code in your research, please cite appropriately and acknowledge the original implementation.

## References

- PyTorch Documentation: https://pytorch.org/
- Biopython: https://biopython.org/
- IMGT Database: http://www.imgt.org/
- NCBI BLAST+: https://blast.ncbi.nlm.nih.gov/

## License

This project is intended for research and educational purposes.

## Contributing

Contributions, suggestions, and forks are welcome for educational and research purposes.
