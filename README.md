# V-Gene Classifier v3.0.0

Deep learning pipeline for automated V-gene discovery and classification in vertebrate genomes using terminal-region encoding and multiclass CNN.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements a complete pipeline for discovering and classifying V-gene segments (immunoglobulin and T-cell receptor variable regions) in vertebrate genomes. The current release (v3_bootstrap3) uses a 2045-dim V3 encoder trained on 60,883 sequences including teleost V-genes, achieving ≥88% TCR recall and ≥95% IG recall across five validated species.

### Key Achievements (v3_bootstrap3)

- **97.2% TRAV recall** on mouse genome (106/109 unique IMGT genes)
- **99.0% IGKV recall** on mouse genome (99/100)
- **Cross-species validated**: human, mouse, ferret, Pongo, Xenopus — same model
- **+20pp human TRAV** vs v2.2 baseline (88.9% vs 68.9%)
- **Out-of-training**: 100% IGHV/IGKV recall on *Pongo pygmaeus*
- **47 FR1 patterns** covering all major IG and TCR V-gene families
- **Fully automated** pipeline: genome → TBLASTN → extract → classify → validate

## 🆕 What's New in v3_bootstrap3 (current)

### Teleost Training Sequences + FR1 Filter Default Change (April 2026)

**Change 1 — Teleost V-gene sequences added to training corpus:**
194 sequences from 4 Actinopterygii species improve cross-phylum TCR generalization:
- 35 TRBV *Oncorhynchus mykiss* (rainbow trout)
- 132 TRAV *Danio rerio* (zebrafish)
- 18 TRAV *Takifugu rubripes* (fugu)
- 9 TRAV *Oncorhynchus mykiss*

**Change 2 — `--require-fr1` default changed to `False`:**
The FR1 pattern filter was on by default in v3_bootstrap2c, causing recall regression
in ferret TRAV/IGKV and completely eliminating teleost/amphibian TRBV candidates.
Default is now OFF for all encoder versions. Enable with `--require-fr1` when processing
well-annotated mammalian genomes and precision is the primary goal.

**Key gains vs v2.2:**
```
Species      Locus   v2.2      v3_bootstrap3   Delta
────────────────────────────────────────────────────
Mouse        TRAV    92.7%     97.2%           +4.5pp
Human        TRAV    68.9%     88.9%           +20.0pp
Ferret       TRAV    76.9%     88.5%           +11.6pp
Ferret       IGHV    92.9%     95.2%           +2.3pp
Pongo        TRAV    92.5%     95.0%           +2.5pp
Xenopus      IGHV    65.8%     71.1%           +5.3pp
```

## 📋 Version History

See [CHANGELOG.md](CHANGELOG.md) for full details.

| Version | Date | Highlights |
|---|---|---|
| **v3_bootstrap3** | Apr 2026 | Teleost TRAV/TRBV training data, `--require-fr1` default=False |
| v3.0.0 | Apr 2026 | RSS-CAC C-term correction, TerminalRegionEncoderV3 (2045-dim), 47 FR1 patterns |
| v2.2.0 | Apr 2026 | 43 FR1 patterns, frame-aware extraction, two-pass TBLASTN workflow |
| v2.1.0 | Apr 2026 | TCR FR1 fix — TRAV/TRBV recall restored cross-species |
| v2.0.0 | Mar 2026 | Terminal-region encoding, hard negatives, first 93%+ recall on mouse |
| v1.x | Jan–Feb 2026 | Initial CNN classifier, multi-species training, IMGT validation |

## ✨ Key Features

- **Complete automation**: Genome → TBLASTN → Extract → Classify → Validate
- **High accuracy**: 93% recall with 99.8% precision
- **Locus classification**: Automatic IGHV/IGKV/TRAV/TRBV identification
- **Robust to noise**: Hard negative training prevents false positives
- **Scalable**: Handles genomes of any size
- **GPU accelerated**: CUDA support for fast training/inference
- **Reproducible**: Fixed seeds, versioned data, documented parameters
- **Publication-ready**: Comprehensive validation and metrics

## 📊 Validation Results (v3_bootstrap3 — Five Species)

**Model:** v3_bootstrap3 — v3 encoder (2045-dim), trained on 60,883 sequences (mammalian + teleost).  
Full metrics history and methodology in [`results/METRICS_HISTORY.md`](results/METRICS_HISTORY.md).

### Mouse (*Mus musculus* — GRCm39)
```
Locus    Recall    Precision    Unique Found
───────────────────────────────────────────────
IGHV     95.3%     93.4%        325/341
IGKV     99.0%     86.5%         99/100
TRAV     97.2%     95.5%        106/109
TRBV     95.5%    100.0%         21/22
───────────────────────────────────────────────
```

### Human (GRCh38)
```
Locus    Recall    Precision    Unique Found
───────────────────────────────────────────────
IGHV     96.1%     91.0%         49/51
IGKV     97.6%     92.4%         41/42
TRAV     88.9%     84.0%         40/45
TRBV     85.4%     94.1%         41/48
───────────────────────────────────────────────
```

### Ferret (*Mustela putorius furo* — GCF_011764305.1)
```
Locus    Recall    Precision    Unique Found
───────────────────────────────────────────────
IGHV     95.2%     82.7%         40/42
IGKV     92.5%     99.8%         37/40
TRAV     88.5%     94.7%         46/52
TRBV     85.0%    100.0%         17/20
───────────────────────────────────────────────
```

### Pongo pygmaeus (*Bornean orangutan* — mPonPyg2) — Out-of-training ✓
```
Locus    Recall    Precision    Unique Found
───────────────────────────────────────────────
IGHV    100.0%     94.0%         59/59
IGKV    100.0%     91.9%         35/35
TRAV     95.0%     66.7%         38/40
TRBV     93.6%     97.9%         44/47
───────────────────────────────────────────────
```

### Xenopus laevis — Out-of-training, non-mammalian ✓
```
Locus    Recall    Precision    Unique Found
───────────────────────────────────────────────
IGHV     71.1%    100.0%         27/38
IGKV      N/A       N/A            —
TRAV      N/A       N/A            —
TRBV     16.7%     42.9%          2/12
───────────────────────────────────────────────
```
No IMGT reference available for IGKV/TRAV in *X. laevis*. TRBV precision drop
relative to v2.2 (100.0% → 42.9%): teleost TRBV sequences added in bootstrap3
share structural features with amphibian TRBV, increasing false-positive rate.

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/py-vgene-classifier.git
cd py-vgene-classifier

# Create conda environment
conda env create -f environment.yml
conda activate vgene

# Or install with pip
pip install -r requirements.txt
```

### Basic Usage

**Complete pipeline (genome → predictions):**
```bash
# 1. Download genome (~5-10 min)
python scripts/04_download_genome.py \
    --accession GCF_000001635.27 \
    --output-dir data/genomes/mouse

# 2. Run TBLASTN (~10-15 min)
python scripts/05_run_tblastn.py \
    --genome data/genomes/mouse/GCF_000001635.27.fna \
    --query data/reference/imgt_mouse/all_vgenes_imgt.fasta \
    --output results/mouse/tblastn_results.txt

# 3. Extract candidates (~5 min)
python scripts/06_extract_candidates.py \
    --tblastn-results results/mouse/tblastn_results.txt \
    --genome data/genomes/mouse/GCF_000001635.27.fna \
    --output results/mouse/candidates.fasta \
    --min-identity 60.0 \
    --clean-terminals \
    --no-merge

# 4. Classify with CNN (~2 min)
python scripts/07_classify_candidates.py \
    --candidates results/mouse/candidates.fasta \
    --model models/v3_bootstrap3/best_model.pt \
    --encoder-version v3 \
    --output results/mouse/vgenes_predicted.fasta

# 5. Validate against IMGT (~3 min)
python scripts/08_validate_predictions.py \
    --predictions results/mouse/vgenes_predicted.fasta \
    --predictions-csv results/mouse/vgenes_predicted_predictions.csv \
    --reference data/reference/imgt_mouse/all_vgenes_imgt.fasta \
    --output-dir results/mouse/validation
```

**Total time: ~25-35 minutes** (mostly automated)

**Output:**
- Predicted V-genes with locus labels (FASTA)
- Per-gene probabilities (CSV)
- Validation metrics and reports

## 📁 Project Structure
```
py-vgene-classifier/
├── data/
│   ├── background/              # Hard negative sequences (~400)
│   ├── background_extended/     # Expanded negatives (~8k)
│   ├── genomes/                 # Downloaded genomes
│   ├── raw/positive/            # IMGT V-gene references
│   └── reference/               # IMGT/NCBI references
├── models/
│   └── v2_multispecies_r3/      # Active model v2.2.0 (129 species, ratio 3:1, run 3)
│       ├── best_model.pt        # weights not tracked in git (*.pt in .gitignore)
│       ├── training_history.csv
│       └── *.png
├── results/
│   ├── mouse_identity60/        # Final results (93% recall)
│   └── bat/                     # Example application
├── scripts/                     # Pipeline scripts (01-09)
├── utils/                       # Utility scripts
├── src/                         # Core modules
│   ├── features/                # Terminal encoding
│   └── models/                  # CNN architecture
├── README.md
├── CHANGELOG.md
├── requirements.txt
└── environment.yml
```

## 🔬 Pipeline Scripts

### Training Pipeline
```
01_generate_background.py       → Generate hard negatives (NCBI)
01b_expand_background.py        → Expand via mutations (optional)
02_prepare_dataset.py           → Prepare training dataset (multi-species)
03_train_model.py               → Train CNN model
```

### Application Pipeline
```
04_download_genome.py           → Download target genome
05_run_tblastn.py               → TBLASTN search
06_extract_candidates.py        → Extract & clean candidates
07_classify_candidates.py       → CNN classification
08_validate_predictions.py      → Validate against IMGT
09_phylogenetic_validation.py   → Tree-based validation (optional)
```

### Utilities
```
utils/explore_vgenes.py         → Data quality checks
utils/parse_imgt_display.py     → Parse IMGT protein display format to FASTA
```

## 🧠 Model Architecture

### CNN_TerminalEncoding (v3, 2045-dim input)

```
Input: (batch, 2045)          # v3 encoder; v2 uses 2000
  -> unsqueeze(1)
  -> (batch, 1, 2045)

Conv1: Conv1d(1, 64, k=5, pad=2) + BatchNorm1d(64) + ReLU + MaxPool1d(2)
Conv2: Conv1d(64, 128, k=5, pad=2) + BatchNorm1d(128) + ReLU + MaxPool1d(2)
Conv3: Conv1d(128, 256, k=5, pad=2) + BatchNorm1d(256) + ReLU + MaxPool1d(2)

Flatten -> (batch, 256 * 255)  # 65,280 for v3 / 64,000 for v2

FC: Linear(65280, 128) + ReLU + Dropout(0.3)
    Linear(128, 64)    + ReLU + Dropout(0.3)
    Linear(64, 5)      # [background, IGHV, IGKV, TRAV, TRBV]

Output: softmax -> class probabilities
```

**Key parameters:**
- **v3 model:** ~8.6M trainable parameters (input 2045)
- **v2 model:** ~8.4M trainable parameters (input 2000)
- **Output:** 5-class probability distribution
- **Regularization:** Batch normalization + Dropout (0.3)

Full architecture specification in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md#model-architecture).

### Terminal-Region Feature Engineering

**N-terminal (40aa):**
- Framework 1 region
- Critical for V-gene identity
- One-hot encoded → 800 features

**C-terminal (40aa):**
- Framework 3 region
- Contains conserved motifs (YYC, YFC, etc.)
- One-hot encoded → 800 features

**Dipeptide frequencies:**
- Captures biochemical composition
- 20×20 = 400 possible dipeptides
- Normalized counts → 400 features

**Why this works:**
- V-genes have conserved terminal regions
- Captures functional constraints
- Robust to CDR length variation
- Computationally efficient

## 📋 Requirements

### Software Dependencies
```
Python 3.11+
BLAST+ 2.10.0+
ClustalO 1.2.4+ (optional, for phylogenetic validation)
CUDA 11.8+ (optional, for GPU acceleration)
```

### Python Packages
```
biopython>=1.81
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### Hardware Recommendations

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB

**Recommended:**
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA GPU with 6+ GB VRAM (RTX 3060 or better)
- Storage: 50+ GB (for multiple genomes)

**Training performance:**
- CPU only: ~4-6 hours
- GPU (RTX 4060): ~30-45 minutes

## 🛠️ Installation Guide

### Option 1: Conda (Recommended)
```bash
# Create environment
conda env create -f environment.yml
conda activate vgene

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: pip + Virtual Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install BLAST+ separately
# Ubuntu/Debian: sudo apt install ncbi-blast+
# macOS: brew install blast
# Windows: Download from NCBI
```

### Verify Installation
```bash
# Check BLAST
makeblastdb -version
tblastn -version

# Check Python packages
python -c "from Bio import SeqIO; import pandas; import torch; print('All OK')"

# Check GPU (optional)
python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "No GPU"
```

## 📖 Detailed Usage

### Training Custom Model

**Step 1: Generate Hard Negatives**
```bash
# Generate from NCBI (~10 min)
python scripts/01_generate_background.py \
    --sources mhc,c_regions,ig_superfamily \
    --target-total 400 \
    --output-dir data/background

# Expand via mutations (optional, ~2 min)
python scripts/01b_expand_background.py \
    --input data/background/background_hard_negatives.fasta \
    --output data/background_extended/background_hard_negatives_8k.fasta \
    --target-total 8000 \
    --mutation-rate 0.02
```

**Step 2: Prepare Dataset**
```bash
python scripts/02_prepare_dataset.py \
    --input-dir data/raw/positive \
    --background data/background/background_hard_negatives.fasta \
    --output-dir data/processed \
    --background-ratio 3.0 \
    --seed 42
```

**Step 3: Train Model**
```bash
python scripts/03_train_model.py \
    --train-csv data/processed/train_multispecies_multiclass.csv \
    --val-csv data/processed/val_multispecies_multiclass.csv \
    --output-dir models/my_model \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.001 \
    --seed 42
```

**Training output:**
- `best_model.pt` - Model weights
- `training_history.csv` - Loss/accuracy per epoch
- `training_history.png` - Training curves
- `confusion_matrix.png` - Validation confusion matrix

### Applying to New Genome

**Step 1: Prepare Genome**
```bash
# Option A: Download from NCBI (auto: tries datasets CLI first, then wget)
python scripts/04_download_genome.py \
    --accession GCF_XXXXXXXXX.X \
    --output-dir data/genomes/species_name \
    --method auto

# Option B: Force direct FTP download (no datasets CLI required)
python scripts/04_download_genome.py \
    --accession GCF_XXXXXXXXX.X \
    --output-dir data/genomes/species_name \
    --method wget

# Option C: Use local genome
# Place genome FASTA in data/genomes/species_name/genome.fna
```

> **Note:** The `wget` method automatically queries the NCBI API to obtain the
> exact assembly name before constructing the FTP URL — no hardcoded suffixes.

**Step 2: Search with TBLASTN**
```bash
python scripts/05_run_tblastn.py \
    --genome data/genomes/species_name/genome.fna \
    --query data/reference/imgt_mouse/all_vgenes_imgt.fasta \
    --output results/species_name/tblastn_results.txt \
    --evalue 1e-5 \
    --threads 8
```

**Parameters:**
- `--evalue`: Lower = more stringent (default: 1e-5)
- `--threads`: Number of CPU cores to use

**Step 3: Extract Candidates**
```bash
python scripts/06_extract_candidates.py \
    --tblastn-results results/species_name/tblastn_results.txt \
    --genome data/genomes/species_name/genome.fna \
    --output results/species_name/candidates.fasta \
    --min-identity 60.0 \
    --extend 150 \
    --clean-terminals \
    --max-vgene-length 120 \
    --no-merge
```

**Key parameters:**
- `--min-identity`: Minimum TBLASTN identity (60-80%)
- `--clean-terminals`: Enable Framework 1 detection (recommended)
- `--max-vgene-length`: Maximum V-gene length for C-terminal trim
- `--no-merge`: Extract each hit separately (recommended for recall)

**Step 4: Classify Candidates**
```bash
python scripts/07_classify_candidates.py \
    --candidates results/species_name/candidates.fasta \
    --model models/v3_bootstrap3/best_model.pt \
    --encoder-version v3 \
    --output results/species_name/vgenes_predicted.fasta \
    --threshold 0.5 \
    --batch-size 64
```

**Threshold selection:**
- `0.5` (default): Balanced precision/recall
- `0.3`: Higher recall, lower precision
- `0.7`: Higher precision, lower recall

**Step 5: Validate Predictions**
```bash
python scripts/08_validate_predictions.py \
    --predictions results/species_name/vgenes_predicted.fasta \
    --predictions-csv results/species_name/vgenes_predicted_predictions.csv \
    --reference data/reference/imgt_species/all_vgenes_imgt.fasta \
    --output-dir results/species_name/validation \
    --min-identity 80 \
    --min-coverage 70
```

**Validation output:**
- Per-locus recall and precision
- Validated predictions (correct locus)
- Misclassified predictions (wrong locus)
- No-match predictions (novel genes?)

### Optional: Phylogenetic Validation
```bash
# Requires ClustalO
conda install -c bioconda clustalo

# Build tree
python scripts/09_phylogenetic_validation.py \
    --imgt data/reference/imgt_species/all_vgenes_imgt.fasta \
    --predictions results/species_name/vgenes_predicted.fasta \
    --predictions-csv results/species_name/vgenes_predicted_predictions.csv \
    --output-dir results/species_name/phylogenetic \
    --top-n 20 \
    --threads 8

# Visualize (requires SeaView or online tool)
seaview results/species_name/phylogenetic/tree.nexus
```

## 🔧 Advanced Configuration

### Identity Threshold Comparison

Test multiple identity thresholds to optimize recall/precision trade-off:
```bash
# High stringency (best precision)
python scripts/06_extract_candidates.py \
    --min-identity 80.0 \
    --output results/species_identity80/candidates.fasta

# Medium stringency (balanced)
python scripts/06_extract_candidates.py \
    --min-identity 70.0 \
    --output results/species_identity70/candidates.fasta

# Low stringency (best recall)
python scripts/06_extract_candidates.py \
    --min-identity 60.0 \
    --output results/species_identity60/candidates.fasta
```

**Expected results (based on mouse validation):**
```
Identity    Recall    Precision    Use case
─────────────────────────────────────────────
≥80%        78%       100%         High confidence only
≥70%        87%       99.8%        Balanced
≥60%        93%       99.8%        Maximum discovery
```

### Model Ensemble

Combine predictions from multiple models for robustness:
```bash
# Train with different seeds
for seed in 42 123 456; do
    python scripts/03_train_model.py \
        --seed $seed \
        --output-dir models/ensemble_${seed}
done

# Classify with each model
# (combine predictions manually or via custom script)
```

### Custom Query Sets

Use species-specific queries for better sensitivity:
```bash
# Extract representative V-genes from IMGT
# Filter by species, functionality, and diversity
# Use as TBLASTN queries
```

## Two-pass TBLASTN for difficult V-gene families

Some V-gene families have TBLASTN hits anchoring in CDR2/FR3 instead of FR1.
The standard 120 aa extraction window misses the gene start. Solution: run a
second TBLASTN pass with `evalue 1e-3` using only the affected families as query.

**Known affected families:**
- **Human TRAV:** TRAV1-1, TRAV1-2, TRAV5, TRAV8-1/2/3/4/6, TRAV16, TRAV23/DV6
- **Mouse TRBV:** TRBV1, TRBV3, TRBV4, TRBV5, TRBV17, TRBV19, TRBV20, TRBV29, TRBV30

**Step 1 — Create family-specific reference FASTA** (already provided in `data/reference/`):
```
data/reference/imgt_human/trav_human_missing.fasta   # 10 TRAV genes
data/reference/imgt_mouse/trbv_mouse_missing.fasta   # 13 TRBV genes
```

**Step 2 — Second TBLASTN pass** (genome DB must already exist from scripts 05):
```bash
python scripts/05_run_tblastn.py \
    --genome data/genomes/<species>/<genome>.fna \
    --query data/reference/<imgt_species>/<locus>_missing.fasta \
    --output results/<species>/tblastn_<locus>_missing.txt \
    --evalue 1e-3 --threads 8 --skip-makedb
```

**Step 3 — Extract candidates from second-pass hits:**
```bash
python scripts/06_extract_candidates.py \
    --tblastn-results results/<species>/tblastn_<locus>_missing.txt \
    --genome data/genomes/<species>/<genome>.fna \
    --output results/<species>/candidates_<locus>_missing.fasta \
    --clean-terminals --min-identity 60
```

**Step 4 — Merge, classify and validate** as usual with scripts 07–08, using the
combined candidates (original + `_missing.fasta`).

**Results with second pass (v2.2.0):**
```
Species   Locus   Standard pass   With second pass   Gain
─────────────────────────────────────────────────────────
Human     TRAV       75.6%           100%*           +24%
Mouse     TRBV       50.0%           95.5%           +45%
```

## 📊 Output Formats

### FASTA Output
```
>candidate_1 query=IGHV1-2*01 chr1:12345-12678(+) len=98 predicted_locus=IGHV prob=0.9987
QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQG...
>candidate_2 query=TRAV14*01 chr14:98765-99012(-) len=101 predicted_locus=TRAV prob=0.9654
AQSVTQSPSSVSAAPGQTAVTINCQSKSSVYNNYLSWFQQKPGQPPKLLIYWASTRESGVPDRFSGS...
```

### CSV Output (with --save-all)
```csv
id,sequence,length,predicted_class,predicted_locus,probability,prob_background,prob_IGHV,prob_IGKV,prob_TRAV,prob_TRBV
candidate_1,QVQLVQ...,98,1,IGHV,0.9987,0.0001,0.9987,0.0003,0.0005,0.0004
candidate_2,AQSVTQ...,101,3,TRAV,0.9654,0.0012,0.0089,0.0045,0.9654,0.0200
```

### Validation Report
```
Per-Locus Breakdown:
───────────────────────────────────────────────────────
Locus    Predictions  Unique Found  Total IMGT  Recall  Precision
───────────────────────────────────────────────────────
IGHV     ~13,200      322           341         94.4%   95.1%
IGKV     ~1,100       97            100         97.0%   95.7%
TRAV     ~1,100       101           109         92.7%   100.0%
TRBV     ~316         21            22          95.5%   100.0%
───────────────────────────────────────────────────────
TOTAL    ~15,700      533           572         93.2%   95.4%
```

## 🔬 Biological Interpretation

### Why V-Genes Are Distinguishable by CNN

V-genes contain characteristic patterns that CNNs can learn:

**1. Framework Regions (FR1-FR3)**
- Conserved amino acid motifs
- Structural constraints for antigen binding
- Typical patterns:
  - **IGHV:** EVQL, QVQL at N-terminus
  - **IGKV/IGLV:** DIQMTQ, DIVMTQ, QSVLTQ
  - **TRAV/TRBV:** Similar to IG but distinct composition

**2. Complementarity-Determining Regions (CDRs)**
- Variable but structurally constrained
- Length and composition signatures
- CDR3 often removed in germline (not used for classification)

**3. Terminal Motifs**
- C-terminal conserved residues (YYC, YFC, YLC)
- Recombination signal sequence (RSS) flanking
- Cysteine positions for disulfide bonds

**4. Biochemical Composition**
- Dipeptide frequencies capture:
  - Hydrophobic/hydrophilic patterns
  - Charged residue distribution
  - Structural propensities

### Locus-Specific Features

**Heavy Chain (IGHV) vs Light Chain (IGKV/IGLV):**
- Length difference (~110-120aa vs ~95-105aa)
- Different N-terminal signatures
- CDR composition patterns
- More hydrophobic residues in IGHV

**T-Cell Receptor (TRAV/TRBV) vs Immunoglobulin (IG):**
- Distinct framework residues
- Different CDR length distributions
- TCR has more charged residues
- Different recombination mechanisms

**Alpha (TRAV) vs Beta (TRBV):**
- TRBV has more diversity in FR2
- Length preferences differ
- Biochemical composition varies

### Why Some Genes Are Difficult

**1. Pseudogenes**
- Frameshifts, stop codons
- Partially degraded sequence
- May lack canonical motifs

**2. Recent Duplications**
- Highly similar paralogs
- Same locus, different genes
- 100% identity possible

**3. Inter-Locus Similarity**
- Some IGHV/IGKV share high identity
- Historical gene conversion events
- Biological confusion, not model error

**4. Strain Differences**
- IMGT is multi-strain (all mouse strains)
- Target genome is single-strain (e.g., C57BL/6J)
- Some genes absent or divergent

## ⚠️ Troubleshooting

### Installation Issues

**Problem: BLAST not found**
```bash
# Solution: Install via conda
conda install -c bioconda blast

# Or check PATH
which makeblastdb
export PATH="/path/to/blast/bin:$PATH"
```

**Problem: GPU not detected**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install correct PyTorch version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Check NVIDIA driver
nvidia-smi
```

**Problem: Out of memory during training**
```python
# In 03_train_model.py, reduce batch size
--batch-size 32  # or 16
```

### Pipeline Issues

**Problem: TBLASTN finds no hits**
```bash
# Solution 1: Lower e-value threshold
--evalue 1e-3  # More permissive

# Solution 2: Check query file format
head data/reference/imgt_mouse/all_vgenes_imgt.fasta

# Solution 3: Verify genome file
file data/genomes/species/genome.fna
# Should be: ASCII text (FASTA format)
```

**Problem: Extract candidates fails**
```bash
# Check TBLASTN output format
head results/species/tblastn_results.txt
# Should have 14 tab-separated columns

# Check contig ID matching
grep ">" data/genomes/species/genome.fna | head -5
# Compare with contig IDs in TBLASTN results
```

**Problem: Low recall on target species**
```bash
# Solution 1: Lower identity threshold
--min-identity 60.0  # or even 50.0 for distant species

# Solution 2: Use broader query set
# Include V-genes from related species

# Solution 3: Disable merging
--no-merge  # Extract each hit separately

# Solution 4: Lower CNN threshold
--threshold 0.3  # More sensitive
```

**Problem: High false positive rate**
```bash
# Solution 1: Increase CNN threshold
--threshold 0.7  # More conservative

# Solution 2: Use stricter identity
--min-identity 80.0  # More stringent

# Solution 3: Enable terminal cleaning
--clean-terminals  # Removes junk sequences
```

**Problem: Validation shows misclassifications**
```bash
# Expected: ~0.2% misclassification rate
# IGHV ↔ IGKV confusion is biological (high similarity)

# Solution: Phylogenetic validation to confirm
python scripts/09_phylogenetic_validation.py ...
```

### Data Issues

**Problem: Missing IMGT reference**
```bash
# Download from IMGT
# 1. Visit http://www.imgt.org/IMGTrepertoire/
# 2. Navigate to Proteins > Protein displays
# 3. Select species and locus
# 4. Copy table to .txt file

# Parse with utility (IMGT protein display format)
python utils/parse_imgt_display.py \
    data/reference/imgt_species/trbv_raw.txt \
    data/reference/imgt_species/trbv.fasta \
    --locus TRBV
```

**Problem: Genome assembly quality issues**
```bash
# Check BUSCO score (should be >90%)
# Check contig N50 (higher is better)
# Fragmented assemblies will have lower recall

# Solution: Use chromosome-level assembly if available
# Or accept lower recall for draft assemblies
```

### Performance Issues

**Problem: Training is very slow**
```bash
# Solution 1: Use GPU
# Ensure CUDA is properly installed

# Solution 2: Reduce dataset size
# Use subset for initial testing

# Solution 3: Increase batch size (if memory allows)
--batch-size 128
```

**Problem: Prediction is slow**
```bash
# Solution: Increase batch size
--batch-size 256  # If GPU memory allows

# Or reduce candidate set
# Use higher identity threshold in extraction
```

## 🎯 Best Practices

### For Maximum Recall
```bash
# Use low identity threshold
--min-identity 60.0

# Don't merge overlapping hits
--no-merge

# Use low CNN threshold
--threshold 0.3

# Enable terminal cleaning
--clean-terminals
```

### For Maximum Precision
```bash
# Use high identity threshold
--min-identity 80.0

# Use high CNN threshold
--threshold 0.7

# Enable terminal cleaning
--clean-terminals
```

### For Balanced Performance
```bash
# Recommended for most use cases
--min-identity 70.0
--threshold 0.5
--clean-terminals
--no-merge
```

## 📚 Methodology

### Training Strategy

**1. Hard Negative Selection**
- Proteins structurally similar to V-genes
- Forces model to learn discriminative features
- Prevents overfitting to simple patterns

**2. Data Augmentation**
- Conservative amino acid substitutions
- Maintains protein structure
- Increases dataset size 20x

**3. Terminal-Region Encoding**
- Focuses on functionally critical regions
- Fixed-length representation
- Sequence-length invariant

**4. Cross-Entropy Loss**
- Suitable for multi-class classification
- Softmax activation for probabilities
- Adam optimizer with learning rate 0.001

### Evaluation Strategy

**1. Stratified Split**
- Maintains class balance
- 80% train, 20% validation
- Fixed random seed for reproducibility

**2. Per-Locus Metrics**
- Recall and precision per locus
- Confusion matrix analysis
- Identifies locus-specific issues

**3. Unique Gene Counting**
- Counts unique IMGT genes found
- Not total prediction count
- More biologically meaningful

**4. Identity Threshold Analysis**
- Tests multiple thresholds (60%, 70%, 80%)
- Establishes recall/precision trade-off
- Informs parameter selection

### Validation Strategy

**1. BLASTP Validation**
- Quantitative assessment
- Per-locus breakdown
- Identifies misclassifications

**2. Phylogenetic Validation**
- Visual confirmation
- Cluster analysis
- Detects systematic errors

**3. Manual Inspection**
- Random sampling of predictions
- Framework region verification
- Terminal motif presence

## 📖 Citation

If you use this pipeline in your research, please cite:
```bibtex
@software{vgene_classifier_v2,
  author = {Luis Raña Cortizo and David N. Olivieri},
  title = {V-Gene Classifier v2.0: Automated Discovery and Classification of Immunoglobulin and T-Cell Receptor V-Genes},
  year = {2026},
  version = {2.1.0},
  url = {https://github.com/yourusername/py-vgene-classifier}
}
```

### Related Publications

This work builds upon methods from:

- Olivieri, D.N., et al. (2019). "Iterative Variable Gene Discovery from Whole Genome Sequencing with a Bootstrapped Multiresolution Algorithm." *Computational and Mathematical Methods in Medicine*. https://doi.org/10.1155/2019/3780245

## 📜 License

This project is licensed for research and educational purposes. See [LICENSE](LICENSE) for details.

**Commercial use requires permission from the authors.**

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Submit a pull request

**Areas for contribution:**
- Support for additional species
- Alternative encoding methods
- Model architecture improvements
- Documentation enhancements
- Bug fixes and optimizations

## 📧 Contact

**Luis Raña Cortizo**
- Email: luisraco95@gmail.com
- Institution: University of Vigo, Intelligent and Adaptive Software Systems PhD Program

## 🙏 Acknowledgments

- **IMGT Database** for V-gene reference sequences
- **NCBI** for genome assemblies and tools
- **PyTorch** team for the deep learning framework
- **Biopython** community for sequence analysis tools
- **CESGA** for computational resources (FinisTerrae III)

## 📚 References

### Tools and Databases

- **IMGT:** http://www.imgt.org/
- **NCBI BLAST+:** https://blast.ncbi.nlm.nih.gov/
- **PyTorch:** https://pytorch.org/
- **Biopython:** https://biopython.org/
- **NCBI Datasets:** https://www.ncbi.nlm.nih.gov/datasets/

### Key Publications

1. Lefranc, M.P., et al. (2015). "IMGT®, the international ImMunoGeneTics information system® 25 years on." *Nucleic Acids Research*, 43(D1), D413-D422.

2. Camacho, C., et al. (2009). "BLAST+: architecture and applications." *BMC Bioinformatics*, 10, 421.

3. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.

4. Cock, P.J., et al. (2009). "Biopython: freely available Python tools for computational molecular biology and bioinformatics." *Bioinformatics*, 25(11), 1422-1423.

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Current version: 3.0.0** (April 2026)

| Version | Date | Highlights |
|---|---|---|
| v3.0.0 | Apr 2026 | RSS-CAC correction, TerminalRegionEncoderV3 (2045 dims), 47 FR1 patterns |
| v2.2.0 | Apr 2026 | FR1 pattern expansion (43 patterns), frame-aware extraction, two-pass TBLASTN |
| v2.1.0 | Apr 2026 | TCR FR1 pattern fix — TRAV/TRBV recall cross-species restored |
| v2.0.0 | Mar 2026 | Terminal-region encoding, hard negatives, 93% recall on mouse |
| v1.3.0 | Feb 2026 | IMGT validation pipeline with BLAST and phylogenetic methods |
| v1.2.0 | Feb 2026 | Multiclass CNN for V-gene locus classification |
| v1.1.0 | Jan 2026 | Multi-species CNN pipeline |
| v1.0.0 | Jan 2026 | Initial V-gene CNN classifier (one-hot encoding) |

---

**Last updated:** April 24, 2026
**Pipeline status:** Production-ready
**Validation:** Cross-species (human, mouse, ferret, Pongo pygmaeus) — IG >=90%, TCR >=83-100%
