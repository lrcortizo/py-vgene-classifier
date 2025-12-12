# Changelog

All notable changes to this project will be documented in this file.

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
