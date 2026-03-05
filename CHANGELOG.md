# Changelog

All notable changes to the V-Gene Classifier project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] - 2026-03-04

### 🎉 Major Release - Complete Pipeline Rewrite

This release represents a fundamental redesign of the V-gene classification pipeline, achieving **93% recall with 99.8% precision** on mouse genome validation.

### Added

#### Core Features
- **Terminal-region encoding**: Fixed-length feature representation (2,000 dimensions)
  - N-terminal 40aa one-hot encoding (800 features)
  - C-terminal 40aa one-hot encoding (800 features)
  - Dipeptide frequency analysis (400 features)
  - Sequence-length invariant architecture

- **Hard negative training**: Biologically relevant negative examples
  - MHC Class I/II proteins (Ig-like domains)
  - Immunoglobulin constant regions
  - Ig superfamily proteins (CD28, CTLA4, ICOS)
  - Data augmentation via conservative mutations (~8,000 sequences)

- **Query ID tracking**: Preserves IMGT gene identity through pipeline
  - Prevents loss in dense genomic regions
  - Enables accurate recall calculation
  - Intelligent deduplication by (query_id, sequence) tuple

- **Automated terminal cleaning**: Framework 1 detection
  - N-terminal motif recognition (EVQL, QVQL, DIQMTQ)
  - C-terminal length normalization (~120aa)
  - Removes upstream and downstream junk sequences

#### Pipeline Scripts (Renumbered)
- `01_generate_background.py`: Generate hard negatives from NCBI
- `01b_expand_background.py`: Expand negatives via mutations (optional)
- `02_prepare_dataset.py`: Prepare training dataset
- `02b_prepare_hybrid_dataset.py`: Create hybrid dataset with hard negatives (optional)
- `03_train_model.py`: Train CNN with terminal encoding
- `04_download_genome.py`: Download genome from NCBI
- `05_run_tblastn.py`: TBLASTN search
- `06_extract_candidates.py`: Extract and clean candidates
- `07_classify_candidates.py`: CNN classification (renamed from filter_positives)
- `08_validate_predictions.py`: Validate against IMGT
- `09_phylogenetic_validation.py`: Phylogenetic tree validation (optional)

#### Utilities
- `utils/explore_vgenes.py`: Data exploration and QC
- `utils/parse_imgt_tables.py`: Parse IMGT reference tables

#### Validation Features
- Per-locus precision and recall metrics
- Unique IMGT gene counting (not just hit counts)
- Identity threshold filtering (60%, 70%, 80%)
- Comprehensive validation reports with per-locus breakdown
- Precision-by-locus analysis

### Changed

#### Architecture
- **Replaced** full-sequence one-hot encoding with terminal-region encoding
- **Reduced** input dimensionality from ~2,320 to 2,000 features
- **Improved** model efficiency and training speed
- **Enhanced** generalization to novel sequences

#### Training
- **Replaced** synthetic random backgrounds with hard negatives
- **Increased** negative class complexity and biological relevance
- **Added** data augmentation via conservative mutations
- **Improved** training data quality and diversity

#### Extraction
- **Changed** deduplication from sequence-only to (query_id, sequence)
- **Added** Framework 1 detection for N-terminal cleaning
- **Added** C-terminal length normalization
- **Improved** candidate quality and downstream accuracy

#### Validation
- **Enhanced** recall calculation using unique IMGT genes
- **Added** precision-by-locus metrics
- **Improved** validation report clarity and detail
- **Added** support for multiple identity thresholds

### Performance Improvements

**Recall (vs v1.3.0):**
```
Locus    v1.3.0    v2.0.0    Improvement
───────────────────────────────────────────
IGHV     ~26%      97.1%     +274%
IGKV     ~46%      96.0%     +109%
TRAV     ~51%      89.0%     +75%
TRBV     ~18%      36.4%     +100%
TOTAL    ~40%      93.0%     +133%
```

**Precision:** 71.8% → 99.8% (+39%)

**Validation on Mouse C57BL/6J (IMGT 572 genes):**
- Found: 532/572 unique genes (93.0% recall)
- Predictions: 16,071 total
- Precision: 99.8% (16 false positives)
- Per-locus recall: IGHV 97.1%, IGKV 96.0%, TRAV 89.0%, TRBV 36.4%

### Fixed
- Query ID loss in dense genomic regions (e.g., 160 IGHV in 2kb cluster)
- False recall inflation from duplicate sequences
- N-terminal junk sequences from TBLASTN extension
- C-terminal over-extension beyond V-region
- Incorrect precision calculation (now per-locus)
- Memory issues with large candidate sets

### Deprecated
- Full-sequence one-hot encoding (replaced by terminal-region)
- Synthetic random background generation (replaced by hard negatives)
- Sequence-only deduplication (replaced by query+sequence)
- Single-threshold validation (now supports multiple thresholds)

### Removed
- `scripts/deprecated/`: Moved obsolete v1.x scripts
- `data/processed/`: Removed old dataset versions
- `data/processed_hybrid/`: Removed intermediate datasets
- `data/processed_v2/`: Removed experimental datasets
- `models/best_model*.pt`: Removed v1.x model files
- `results/`: Cleaned ~78 MB of intermediate results
- `notebooks/`: Removed empty directory
- `tests/`: Removed empty directory

### Project Structure
- Reorganized scripts with logical numbering (01-09)
- Created `utils/` directory for auxiliary tools
- Cleaned `data/` directory (removed ~59 MB)
- Cleaned `models/` directory (removed ~7 MB)
- Cleaned `results/` directory (removed ~78 MB)
- Total cleanup: ~144 MB removed

### Documentation
- Complete README.md rewrite for v2.0.0
- Added detailed pipeline documentation
- Enhanced troubleshooting section
- Added best practices guide
- Updated citation information

---

## [1.3.0] - 2025-02-03

### Added
- IMGT validation pipeline
- Manual reference extraction from IMGT protein displays
- NCBI reference extraction from GFF annotations
- Phylogenetic tree analysis with ClustalO
- `scripts/11_parse_imgt_tables.py`
- `scripts/12_validate_predictions.py`
- `scripts/13_phylogenetic_validation.py`

### Validation Results (Mouse)
- IMGT reference: 79/202 validated (71.8% precision)
- NCBI reference: 87/202 validated (63.0% precision)
- Main challenge: IGHV ↔ IGKV confusion (~15%)
- Phylogenetic analysis: ~70-80% correct clustering

---

## [1.2.0] - 2024-12-28

### Added
- Multiclass CNN for automatic locus classification
- 5-class output: [background, IGHV, IGKV, TRAV, TRBV]
- Confusion matrix visualization
- Per-locus metrics (precision, recall, F1)
- `scripts/05_prepare_multispecies_dataset_multiclass.py`
- `scripts/06_train_multispecies_cnn_multiclass.py`
- `scripts/10_filter_positives_multiclass.py`

### Performance
- Training accuracy: 99.99%
- Validation accuracy: 99.99%
- F1 scores: 0.9991-1.0000 per class
- Training time: ~1 hour (RTX 4060)

### Changed
- CrossEntropyLoss for multi-class classification
- Enhanced dataset preparation with locus parsing

---

## [1.1.0] - 2024-11-22

### Added
- Multi-species training support
- Complete genome-to-prediction pipeline
- Synthetic background generation
- `scripts/05_prepare_multispecies_dataset.py`
- `scripts/05b_generate_synthetic_background.py`
- `scripts/06_train_multispecies_cnn.py`
- `scripts/07_download_genome.py`
- `scripts/08_run_tblastn.py`
- `scripts/09_extract_candidates.py`
- `scripts/10_filter_positives.py`

### Performance
- Significant improvement over single-species models
- Multi-species model finds substantially more V-genes

---

## [1.0.0] - 2024-11-15

### Initial Release
- Basic CNN for V-gene classification
- Binary classification (V-gene vs background)
- Single-species training (human)
- One-hot encoding of full sequences
- `scripts/01_explore_vgenes.py`
- `scripts/02_download_background.py`
- `scripts/03_prepare_dataset.py`
- `scripts/04_train_classifier.py`

### Features
- GPU acceleration support
- Basic training/validation pipeline
- Model saving and loading

---

## Version Format

`[MAJOR.MINOR.PATCH]` - Date

- **MAJOR**: Incompatible API changes or complete rewrites
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

## Links

- [GitHub Repository](https://github.com/yourusername/py-vgene-classifier)
- [Issues](https://github.com/yourusername/py-vgene-classifier/issues)
- [Documentation](README.md)
