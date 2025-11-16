# V-Gene Classifier

Binary classifier to identify adaptive immune system V genes using PyTorch and CNNs.

## Project Overview

This project implements a Convolutional Neural Network (CNN) to distinguish V gene segments from genomic background sequences. The classifier uses one-hot encoded amino acid sequences and achieves high performance through pattern recognition of characteristic V-gene motifs.

## Key Features

- **Deep Learning Architecture**: 1D CNN optimized for protein sequence classification
- **One-Hot Encoding**: Efficient representation of amino acid sequences
- **Modular Design**: Reusable components for encoding, model architecture, and training
- **Complete Pipeline**: From raw FASTA files to trained model
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Reproducible Results**: Fixed random seeds and documented parameters

## Project Structure
```
py-vgene-classifier/
│
├── data/
│   ├── raw/
│   │   ├── positive/          # V gene FASTA files
│   │   └── negative/          # Background sequences
│   └── processed/             # Train/val splits (CSV + FASTA)
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
│   ├── 03_prepare_dataset.py      # Train/val split
│   ├── 04_train_classifier.py     # Model training
│   ├── verify_split.py            # Data quality checks
│   └── inspect_predictions.py     # Prediction analysis
│
├── models/
│   └── best_model.pt          # Trained model weights
│
├── results/
│   ├── training_history.png   # Training curves
│   ├── training_history.csv   # Metrics log
│   └── vgene_stats.csv        # Dataset statistics
│
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests (optional)
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites
- Python 3.11+

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Create conda environment
conda create -n vgene_classifier python=3.11 -y
conda activate vgene_classifier

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install biopython pandas numpy matplotlib scikit-learn python-dotenv
```

#### Option 2: Using venv (Alternative)
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

#### Option 3: Direct pip install
```bash
# Install dependencies directly (requires Python 3.11+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (use `.env.example` as template):
```
NCBI_EMAIL=your.email@example.com
```

This is required for downloading background sequences from NCBI.

## Data Requirements

### Positive Class (V Genes)
Place your V gene FASTA files in `data/raw/positive/`. The system expects amino acid sequences in standard FASTA format:
```
>sequence_id_1
QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMH...
>sequence_id_2
DIQMTQSPSSLSASVGDRVTITCRASQSISSWLA...
```

### Negative Class (Background)
Background sequences can be:
1. **Auto-generated**: Run `02_download_background.py` to download and translate random genomic regions
2. **User-provided**: Place your background sequences in `data/raw/negative/background.fasta`

The background should have similar length distribution to positive sequences for fair classification.

## Usage

### 1. Explore V Gene Data
```bash
python scripts/01_explore_vgenes.py
```
**Output**: 
- Statistics of V gene sequences (count, length distribution)
- `results/vgene_stats.csv`

### 2. Download Background Sequences (Optional)
```bash
python scripts/02_download_background.py
```
**Output**: 
- Translated genomic sequences in `data/raw/negative/background.fasta`
- Adjustable ratio to positive sequences

**Note**: Requires NCBI_EMAIL in `.env` file.

### 3. Prepare Dataset
```bash
python scripts/03_prepare_dataset.py
```
**Output**: 
- Train/val splits (80/20) in `data/processed/`
- Both CSV and FASTA formats
- Stratified split maintaining class ratios

### 4. Train Model
```bash
python scripts/04_train_classifier.py
```
**Output**: 
- Trained model: `models/best_model.pt`
- Training plots: `results/training_history.png`
- Metrics log: `results/training_history.csv`

### 5. Verify Data Quality
```bash
python scripts/verify_split.py
```
**Output**:
- Checks for duplicate sequences
- Verifies train/val overlap
- Length distribution statistics

### 6. Inspect Predictions
```bash
python scripts/inspect_predictions.py
```
**Output**:
- Sample predictions with probabilities
- Model confidence analysis
- Per-class accuracy breakdown

## Methodology

### Feature Representation
- **Encoding**: One-hot encoding (20-dimensional binary vector per amino acid)
- **Sequence length**: Fixed maximum length with zero-padding for shorter sequences
- **Input shape**: `(batch_size, 20, sequence_length)`

### Model Architecture
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
Output (batch, 1) - probability of V-gene
```

**Key Components**:
- **Convolutional layers**: Detect local sequence motifs
- **Batch normalization**: Stabilize training
- **Max pooling**: Reduce dimensionality and provide translation invariance
- **Dropout**: Prevent overfitting
- **Sigmoid activation**: Output probabilities (0-1)

### Training Configuration

Default hyperparameters (configurable in `scripts/04_train_classifier.py`):
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
OPTIMIZER = Adam
LOSS = Binary Cross Entropy
```

### Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity / True positive rate
- **F1 Score**: Harmonic mean of precision and recall (primary metric)
- **AUC-ROC**: Area under receiver operating characteristic curve

## Customization

### Adjusting Model Architecture

Edit `src/models/classifier.py`:
```python
model = VGeneCNN(
    input_channels=20,          # Number of amino acids
    seq_length=116,             # Maximum sequence length
    num_filters=[64, 128, 256], # Filters per conv layer
    kernel_size=3,              # Convolutional kernel size
    dropout=0.3                 # Dropout probability
)
```

### Changing Dataset Ratio

Edit `scripts/02_download_background.py`:
```python
NUM_SEQUENCES = 753  # Number of background sequences
# Adjust based on your positive class size
```

### Modifying Train/Val Split

Edit `scripts/03_prepare_dataset.py`:
```python
train_df, val_df = train_test_split(
    df,
    test_size=0.2,      # Change validation percentage
    random_state=42,    # Change seed for different split
    stratify=df['label']
)
```

## Understanding Results

### Training Curves

The `training_history.png` plot shows:
1. **Loss curves**: Should decrease over epochs
2. **Accuracy**: Should increase and stabilize
3. **F1 Score**: Primary metric for imbalanced datasets
4. **AUC-ROC**: Measure of class separability

### Good Training Signs
- ✅ Train loss decreases consistently
- ✅ Val loss decreases and stabilizes
- ✅ Small gap between train and val metrics
- ✅ F1 and AUC reach high values (>0.9)

### Warning Signs
- ⚠️ Val loss increases while train loss decreases → Overfitting
- ⚠️ Both losses remain high → Underfitting
- ⚠️ Large gap between train and val → Generalization issues

## Biological Interpretation

### Why V Genes Are Distinguishable

V genes typically contain:
1. **Recombination Signal Sequences (RSS)**: Characteristic motifs
2. **Framework regions**: Conserved amino acid patterns
3. **CDR loops**: Variable but structurally defined regions
4. **Immunoglobulin/TCR domains**: Specific protein architecture

The CNN learns to detect these patterns through its convolutional filters.

### Model Confidence

Examine prediction probabilities using `inspect_predictions.py`:
- **High confidence (>0.9)**: Strong signal detected
- **Medium confidence (0.5-0.9)**: Ambiguous cases
- **Low confidence (<0.5)**: Likely background

## Extending the Project

### For More Challenging Classification

1. **Use similar negative class**: Instead of random genomic sequences, use:
   - Immunoglobulin constant regions
   - Other immune-related proteins
   - V-gene pseudogenes

2. **Cross-species validation**: Train on one species, test on another

3. **Sequence augmentation**: Add mutations to simulate natural variation

4. **Multi-class classification**: Distinguish between V gene subtypes (IGHV, IGKV, etc.)

## Troubleshooting

### Common Issues

**1. NCBI Download Fails**
```bash
# Check your .env file exists and contains valid email
cat .env
# Should show: NCBI_EMAIL=your.email@example.com
```

**2. Out of Memory During Training**
```python
# Reduce batch size in 04_train_classifier.py
BATCH_SIZE = 16  # Instead of 32
```

**3. Poor Performance**
- Check data quality with `verify_split.py`
- Ensure positive and negative sequences have similar length distributions
- Verify no data leakage between train and val

**4. Import Errors**
```bash
# Ensure you're in the project root and environment is activated
cd py-vgene-classifier
conda activate vgene_classifier  # or: source venv/bin/activate
```

## Files Generated

- `models/best_model.pt` - Trained CNN weights (PyTorch state dict)
- `results/training_history.png` - Loss, accuracy, F1, AUC curves
- `results/training_history.csv` - Per-epoch metrics
- `data/processed/*.csv` - Train/val datasets with labels
- `data/processed/*.fasta` - Train/val in FASTA format

## Technical Details

### Dependencies

- **torch**: Neural network framework
- **biopython**: FASTA parsing and sequence manipulation
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Plotting
- **scikit-learn**: Metrics and train/test split
- **python-dotenv**: Environment variable management

### Hardware Requirements

- **Minimum**: 4GB RAM, modern CPU
- **Recommended**: 8GB+ RAM
- **GPU**: Optional (CPU version is used by default)

For GPU support, install PyTorch with CUDA:
```bash
# Check CUDA version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Contributing

This is a research project. Contributions, suggestions, and forks are welcome for educational purposes.

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Biopython Tutorial: https://biopython.org/wiki/Documentation
- NCBI Entrez API: https://www.ncbi.nlm.nih.gov/books/NBK25501/

## License

This project is intended for research and educational purposes.

## Citation

If you use this code in your research, please cite appropriately and acknowledge the original implementation.