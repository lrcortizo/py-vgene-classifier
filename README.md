# V-Gene Classifier - Multi-Species CNN

Deep learning pipeline for V-gene discovery in vertebrate genomes using multi-species trained CNN with locus-specific classification.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to identify and classify V gene segments (variable genes from immunoglobulin and T-cell receptor loci) in vertebrate genomes. The classifier uses one-hot encoded amino acid sequences and achieves high performance through pattern recognition of characteristic V-gene motifs.

**NEW in v1.2.0**: Multiclass CNN for automatic locus classification (IGHV, IGKV, TRAV, TRBV), achieving 99.99% validation accuracy.

**v1.1.0**: Multi-species training pipeline with complete genome-to-prediction workflow.

## Key Features

- **Multiclass Classification** (v1.2.0): Automatic identification of V-gene locus (IGHV, IGKV, TRAV, TRBV)
- **Multi-Species Training**: Supports training on V-genes from multiple vertebrate species
- **Complete Pipeline**: From genome download to final V-gene predictions with locus assignment
- **GPU Acceleration**: CUDA support for faster training and inference
- **Synthetic Background**: Automated generation of negative training examples
- **TBLASTN Integration**: Candidate identification from whole genomes
- **High Accuracy**: Achieves 99.99% validation accuracy on multi-species datasets
- **Modular Design**: Reusable components for encoding, model architecture, and training
- **Reproducible Results**: Fixed random seeds and documented parameters

## What's New in v1.3.0

### IMGT Validation Pipeline
- **Dual validation approach**: Quantitative (BLAST) + Visual (phylogenetic tree)
- **IMGT reference integration**: Manual extraction from IMGT protein displays
- **NCBI reference extraction**: Automated parsing from genome GFF annotations
- **Phylogenetic tree analysis**: ClustalO MSA + SeaView visualization

### Validation Scripts
```bash
# Parse IMGT protein display tables
scripts/11_parse_imgt_tables.py

# BLAST-based validation with metrics
scripts/12_validate_predictions.py

# Phylogenetic tree validation (Olivieri's method)
scripts/13_phylogenetic_validation.py
```

### Validation Results (Mouse GRCm39)

**IMGT Reference (572 functional V-genes):**
- Precision: 71.8% (of matched predictions)
- Validated: 79/202 (39.1%)
- Challenge: IGHV ↔ IGKV confusion (~15%)

**NCBI Reference (646 annotated V-genes):**
- Precision: 63.0% (relaxed criteria)
- Validated: 87/202 (43.1%)

**Phylogenetic Analysis:**
- 652 sequences aligned (572 IMGT + 80 top predictions)
- ~70-80% predictions cluster correctly with IMGT
- Visual confirmation of BLAST results

### Key Findings
- Model successfully separates IG vs TR gene families
- Main challenge: Heavy vs light chain distinction (IGHV vs IGKV)
- Some genes show 100% identity across loci (biological limitation)
- Suitable for phylogenetic analysis at family level

## What's New in v1.2.0

### Multiclass Classification
- **5-class CNN**: Classifies V-genes by locus (IGHV, IGKV, TRAV, TRBV) + background
- **99.99% accuracy**: On validation set of 30,301 sequences
- **Per-locus metrics**: Individual precision/recall for each class
- **Automatic classification**: No BLAST post-processing required for locus identification
- **Probability output**: Returns confidence scores for all 5 classes

### Technical Improvements
- Trained on 113,691 V-genes from 50+ vertebrate species
- CrossEntropyLoss with softmax for multi-class classification
- Confusion matrix visualization for model evaluation
- Enhanced dataset preparation with locus label parsing

### Model Performance
- Training accuracy: 99.99%
- Validation F1 score: 0.9999
- Convergence: ~37 epochs
- Training time: ~1 hour on RTX 4060 GPU

## What's New in v1.1.0

### Major Improvements
- **Multi-species dataset support**: Train on V-genes from diverse vertebrate species
- **Complete discovery pipeline**: End-to-end workflow from genome to predictions
- **Synthetic background generation**: Fast, reproducible negative training examples
- **Significant improvement**: Multi-species model finds substantially more V-genes than single-species models

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
│       ├── classifier.py      # CNN architecture (multiclass)
│       └── train.py           # Training/evaluation functions
│
├── scripts/
│   ├── 01_explore_vgenes.py                    # Data exploration
│   ├── 02_download_background.py               # NCBI background download
│   ├── 03_prepare_dataset.py                   # Train/val split (single-species)
│   ├── 04_train_classifier.py                  # Model training (single-species)
│   ├── 05_prepare_multispecies_dataset.py      # Multi-species dataset (binary)
│   ├── 05_prepare_multispecies_dataset_multiclass.py  # Multi-species dataset (multiclass)
│   ├── 05b_generate_synthetic_background.py    # Synthetic negatives
│   ├── 06_train_multispecies_cnn.py            # Multi-species training (binary)
│   ├── 06_train_multispecies_cnn_multiclass.py # Multi-species training (multiclass)
│   ├── 07_download_genome.py                   # Genome download
│   ├── 08_run_tblastn.py                       # TBLASTN search
│   ├── 09_extract_candidates.py                # Candidate extraction
│   ├── 10_filter_positives.py                  # CNN filtering (binary)
│   ├── 10_filter_positives_multiclass.py       # CNN filtering (multiclass)
│   ├── 11_parse_imgt_tables.py                 # Parse IMGT references
│   ├── 12_validate_predictions.py              # BLAST validation
│   ├── 13_phylogenetic_validation.py           # Tree-based validation
│   ├── verify_split.py                         # Data quality checks
│   └── inspect_predictions.py                  # Prediction analysis
│
├── models/
│   ├── best_model.pt              # Single-species trained model
│   ├── best_model_multispecies.pt # Multi-species trained model (binary)
│   └── best_model_multiclass.pt   # Multi-species trained model (multiclass)
│
├── results/
│   ├── training_history.png
│   ├── training_history_multispecies.png
│   ├── training_history_multiclass.png
│   └── confusion_matrix_multiclass.png
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
- ClustalO (1.2.4+) - for phylogenetic validation (optional)
- SeaView (optional, for tree visualization)
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

## Quick Start - Multiclass Pipeline (v1.2.0)

### Complete V-Gene Discovery with Locus Classification
```bash
# 1. Generate synthetic background (~10 seconds)
python scripts/05b_generate_synthetic_background.py

# 2. Prepare multiclass dataset (~2 minutes)
python scripts/05_prepare_multispecies_dataset_multiclass.py \
    --input-dir /path/to/annotated_vgenes \
    --loci ighv igkv trav trbv \
    --output-dir data/processed

# 3. Train multiclass CNN (~1 hour with GPU)
python scripts/06_train_multispecies_cnn_multiclass.py \
    --train-csv data/processed/train_multispecies_multiclass.csv \
    --val-csv data/processed/val_multispecies_multiclass.csv \
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

# 7. Classify with multiclass CNN (~1-2 minutes)
python scripts/10_filter_positives_multiclass.py \
    --candidates results/target/candidates.fasta \
    --model models/best_model_multiclass.pt \
    --output results/target/vgenes_predicted.fasta \
    --threshold 0.5 \
    --save-all
```

**Output**: FASTA with predicted V-genes annotated by locus (IGHV, IGKV, TRAV, TRBV)

**Total time**: ~2-3 hours (mostly automated)

## Validation Pipeline (v1.3.0)

### Validate Predictions Against IMGT

**Step 1: Extract IMGT References**

Manually download V-gene sequences from IMGT:
1. Visit IMGT Repertoire: http://www.imgt.org/IMGTrepertoire/
2. Navigate to Proteins > Protein displays
3. Select species (e.g., Mus musculus) and locus (IGHV, IGKV, TRAV, TRBV)
4. Copy complete table to text files
```bash
# Parse IMGT tables to FASTA
python scripts/11_parse_imgt_tables.py \
    --input-dir data/reference/imgt_mouse \
    --output-dir data/reference/imgt_mouse
```

**Step 2: BLAST Validation**
```bash
# Combine IMGT references
cat data/reference/imgt_mouse/*_mouse_imgt.fasta > \
    data/reference/imgt_mouse/all_vgenes_imgt.fasta

# Run BLAST validation
python scripts/12_validate_predictions.py \
    --predictions results/mouse/vgenes_predicted.fasta \
    --predictions-csv results/mouse/vgenes_predicted_all_predictions.csv \
    --reference data/reference/imgt_mouse/all_vgenes_imgt.fasta \
    --output-dir results/mouse/validation_imgt \
    --min-identity 80 \
    --min-coverage 70
```

**Output**: CSVs with validated, misclassified, and no-match predictions

**Step 3: Phylogenetic Tree Validation**
```bash
# Requires ClustalO
conda install -c bioconda clustalo

# Build phylogenetic tree
python scripts/13_phylogenetic_validation.py \
    --imgt data/reference/imgt_mouse/all_vgenes_imgt.fasta \
    --predictions results/mouse/vgenes_predicted.fasta \
    --predictions-csv results/mouse/vgenes_predicted_all_predictions.csv \
    --output-dir results/mouse/phylogenetic_validation \
    --top-n 20 \
    --threads 8
```

**Output**:
- Multiple sequence alignment (FASTA)
- Phylogenetic tree (Newick and Nexus formats)
- Ready for visualization in SeaView

**Step 4: Visualize Tree**
```bash
# Open in SeaView (if installed)
seaview results/mouse/phylogenetic_validation/tree.nexus

# Or use online tools:
# - iTOL: https://itol.embl.de/
# - Phylo.io: https://phylo.io/
```

**Interpretation**:
- Predictions clustering with IMGT of same locus → Validated
- Predictions clustering with IMGT of different locus → Misclassified
- Predictions isolated from all IMGT → Potential false positives

**Total validation time**: ~30-60 minutes (mostly alignment)

## Data Requirements

### Multiclass Pipeline

**Input**: Annotated V-gene FASTAs from multiple species with locus information
- Format: Headers must contain locus identifier (e.g., `>ID|Species|ighv`)
- Source: IMGT alignment, custom annotation pipeline
- Recommended loci: IGHV, IGKV, TRAV, TRBV

**Queries**: Representative V-genes for TBLASTN
- Generate 50-200 sequences per locus from training dataset
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

### CNN Design (v1.2.0)
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
FC(64→5) + Softmax
  ↓
Output: Class probabilities [background, IGHV, IGKV, TRAV, TRBV]
```

**Parameters**: 595,525 trainable
**Input**: One-hot encoded protein sequences (20 amino acids × sequence length)
**Output**: 5-class probability distribution

### Performance

**Multiclass model (v1.2.0):**
- Overall accuracy: 99.99%
- Per-class F1 scores:
  - Background: 1.0000
  - IGHV: 0.9999
  - IGKV: 0.9998
  - TRAV: 0.9999
  - TRBV: 0.9991

**Binary model (v1.1.0):**
- Training accuracy: ~99-100%
- Validation accuracy: ~97-100%
- F1 Score: ~0.97-1.0

## Methodology

### Feature Representation
- **Encoding**: One-hot (20-dimensional per amino acid)
- **Max length**: 116 amino acids (configurable)
- **Padding**: Zero-padding for shorter sequences

### Training Configuration
```python
# Multiclass (v1.2.0)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
OPTIMIZER = Adam
LOSS = CrossEntropyLoss

# Binary (v1.1.0)
LOSS = Binary Cross Entropy
```

### Evaluation Metrics
- Accuracy, Precision, Recall (per class)
- F1 Score (weighted average)
- Confusion Matrix

## Output Format

### Multiclass Predictions

**FASTA output:**
```
>candidate_1 predicted_locus=IGHV prob=1.0000
QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMH...
>candidate_2 predicted_locus=TRAV prob=0.9987
DIQMTQSPSSLSASVGDRVTITCRASQSISSWLA...
```

**CSV output** (with `--save-all`):
```
id,sequence,length,predicted_locus,probability,prob_background,prob_IGHV,prob_IGKV,prob_TRAV,prob_TRBV
candidate_1,QVQLVQ...,95,IGHV,1.0000,0.0000,1.0000,0.0000,0.0000,0.0000
candidate_2,DIQMTQ...,92,TRAV,0.9987,0.0001,0.0002,0.0001,0.9987,0.0009
```

## Customization

### Adjust CNN Threshold
```bash
python scripts/10_filter_positives_multiclass.py \
    --threshold 0.3  # Lower = more sensitive
```

### Modify TBLASTN Sensitivity
```bash
python scripts/08_run_tblastn.py \
    --evalue 1e-3  # More permissive
```

## Biological Interpretation

### Why V Genes Are Distinguishable

V genes contain:
1. **Framework regions (FR)**: Conserved amino acid patterns
2. **CDR loops**: Variable but structurally constrained
3. **Recombination signal sequences**: Characteristic motifs
4. **Immunoglobulin/TCR domains**: Specific architecture

### Locus-Specific Features

The multiclass CNN learns to distinguish:
- **IGHV vs IGKV**: Heavy vs light chain immunoglobulins
- **TRAV vs TRBV**: Alpha vs beta chain T-cell receptors
- **IG vs TR**: B-cell vs T-cell receptor genes

## Troubleshooting

### BLAST Not Found
```bash
conda install -c bioconda blast
```

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
```python
BATCH_SIZE = 32  # Reduce in training scripts
```

### Low Performance on Target Species
- Include more diverse species in training
- Lower CNN threshold (0.3-0.4)
- Use species-specific queries
- Check genome assembly quality

### ClustalO Not Found
```bash
# Install with conda (recommended)
conda install -c bioconda clustalo

# Or compile from source
# http://www.clustal.org/omega/
```

### SeaView Not Available
```bash
# Ubuntu/Debian
sudo apt install seaview

# macOS
brew install seaview

# Windows: Download from http://doua.prabi.fr/software/seaview

# Or use online tree viewers:
# - iTOL: https://itol.embl.de/
# - Phylo.io: https://phylo.io/
```

### Validation Shows Low Recall
- Normal for comprehensive references (IMGT has 300+ genes per locus)
- Conservative TBLASTN e-value (1e-5) limits sensitivity
- Strain differences between training data and target genome
- Consider: Focus on precision rather than recall for phylogenetic studies

## Citation

If you use this pipeline in your research, please cite appropriately.

## References

- PyTorch: https://pytorch.org/
- Biopython: https://biopython.org/
- IMGT Database: http://www.imgt.org/
- NCBI BLAST+: https://blast.ncbi.nlm.nih.gov/

## License

This project is intended for research and educational purposes.

## Contributing

Contributions and suggestions are welcome for educational and research purposes.
