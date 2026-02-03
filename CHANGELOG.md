# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2024-12-28

### Added
- **Multiclass CNN architecture**: 5-class classification (background, IGHV, IGKV, TRAV, TRBV)
- Multiclass dataset preparation script (`05_prepare_multispecies_dataset_multiclass.py`)
- Multiclass training script (`06_train_multispecies_cnn_multiclass.py`)
- Multiclass inference script (`10_filter_positives_multiclass.py`)
- Confusion matrix visualization for multiclass evaluation
- Per-locus probability outputs in CSV format
- Support for locus label parsing from annotated FASTA headers
- Recursive file search in dataset preparation

### Changed
- CNN architecture modified for 5-class output (softmax instead of sigmoid)
- Loss function changed from Binary Cross Entropy to CrossEntropyLoss
- Training evaluation now includes per-class metrics
- Output format includes predicted locus and all class probabilities
- Updated .gitignore to include v1.2.0 artifacts

### Performance
- Training accuracy: 99.99% (151,505 sequences)
- Validation accuracy: 99.99% (30,301 sequences)
- F1 score: 0.9999 (weighted average)
- Convergence: epoch 37
- Per-class precision: >99.88% for all classes
- Training time: ~1 hour on RTX 4060 GPU

### Results - Mouse Genome (GRCm39)
- Total candidates identified: 462
- V-genes predicted: 202
  - IGHV: 101 genes
  - IGKV: 21 genes
  - TRAV: 56 genes
  - TRBV: 24 genes
- Median prediction confidence: 1.0

### Dataset
- Total sequences: 151,505
  - V-genes: 113,691 (IGHV: 41,736, IGKV: 25,531, TRAV: 38,334, TRBV: 8,090)
  - Background: 37,814 (synthetic)
- Train/val split: 121,204 / 30,301 (80/20)
- Species: 50+ vertebrates (mammals and reptiles)

### Technical Improvements
- Automatic locus classification (no BLAST post-processing needed)
- Softmax output with probability distribution across all classes
- Enhanced error messages and progress reporting
- Comprehensive classification report with per-class metrics

## [1.1.0] - 2024-12-13

### Added
- Multi-species training dataset support
- Synthetic background generation script (`05b_generate_synthetic_background.py`)
- Complete V-gene discovery pipeline (scripts 05-10):
  - `05_prepare_multispecies_dataset.py` - Multi-species dataset preparation
  - `06_train_multispecies_cnn.py` - Multi-species CNN training
  - `07_download_genome.py` - Automated genome download
  - `08_run_tblastn.py` - TBLASTN search integration
  - `09_extract_candidates.py` - Candidate extraction with translation
  - `10_filter_positives.py` - CNN-based filtering
- GPU acceleration support (CUDA)
- Comprehensive CSV output with prediction probabilities

### Changed
- Upgraded CNN training to support multi-species datasets
- Improved sequence extraction with multiple reading frame translation
- Enhanced ID parsing for NCBI genome assemblies
- Better error handling and progress reporting throughout pipeline

### Performance
- Achieved 100% accuracy on multi-species validation set
- Significant improvement in V-gene discovery (multi-species vs single-species)
- Processing time: ~2-3 hours for complete pipeline (training + inference)

### Fixed
- Contig ID mismatch between TBLASTN and genome FASTA files
- Translation frame handling for candidate extraction
- Memory efficiency in large-scale TBLASTN processing

## [1.0.0] - 2024-12-06

### Added
- Initial CNN architecture for V-gene classification
- Single-species training dataset support
- Basic TBLASTN pipeline
- Model training and evaluation scripts (01-04)
- One-hot encoding for protein sequences
- Training visualization and metrics

### Performance
- 100% accuracy on single-species V-gene dataset
