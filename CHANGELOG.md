# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0] - 2025-02-03

### Added
- **IMGT validation pipeline** following Olivieri's methodology:
  - `11_parse_imgt_tables.py` - Parse IMGT protein display tables to FASTA
  - `12_validate_predictions.py` - BLAST-based validation with quantitative metrics
  - `13_phylogenetic_validation.py` - Phylogenetic tree validation with ClustalO + SeaView
- IMGT reference extraction for mouse (572 functional V-genes)
- NCBI GFF-based reference extraction (646 annotated V-genes)
- Dual validation approach: quantitative (BLAST) + visual (phylogenetic tree)
- Validation summary documentation (`VALIDATION_SUMMARY.md`)

### Validation Results - Mouse GRCm39

#### IMGT Reference (572 functional V-genes)
**Method:** BLASTP with ≥80% identity, ≥70% coverage
- Validated: 79/202 (39.1%)
- Misclassified: 31/202 (15.3%)
- No IMGT match: 92/202 (45.5%)
- **Precision (of matched): 71.8%**

**Recall by locus:**
- IGHV: 10.3% (35/341 IMGT genes found)
- IGKV: 12.0% (12/100 IMGT genes found)
- TRAV: 23.9% (26/109 IMGT genes found)
- TRBV: 27.3% (6/22 IMGT genes found)

#### NCBI Reference (646 annotated V-genes from GFF)
**Strict criteria** (≥95% identity, ≥90% coverage):
- Validated: 8/202 (4.0%)
- High-quality matches: 15/995 BLAST hits

**Relaxed criteria** (≥80% identity, ≥70% coverage):
- Validated: 87/202 (43.1%)
- Misclassified: 51/202 (25.2%)
- **Precision (of matched): 63.0%**
- Average recall: ~20% across loci

#### Phylogenetic Tree Analysis
**Method:** ClustalO MSA + tree building
- Total sequences: 652 (572 IMGT + 80 top predictions)
- Multiple sequence alignment completed
- Trees generated: Newick and Nexus formats
- Visualization: SeaView-compatible
- **Result:** ~70-80% predictions cluster correctly with IMGT of same locus
- Visual confirmation of quantitative BLAST results

### Key Findings

#### Model Performance
- Successfully identifies V-genes with high specificity
- Strong separation between IG and TR gene families
- IG vs TR classification: robust and reliable

#### Main Challenge: IGHV vs IGKV Confusion
- ~15% misclassification between heavy and light chain immunoglobulins
- Root cause: High sequence similarity (some genes show 100% identity across loci)
- Biological limitation: genes share recent evolutionary origin
- Examples: candidate_222, candidate_342 (both 100% identity to wrong locus)

#### Comparison: IMGT vs NCBI
|Metric|IMGT|NCBI (relaxed)|
|------|----|----|
|Precision|71.8%|63.0%|
|Validated|79 (39.1%)|87 (43.1%)|
|Recall (IGHV)|10.3%|21.3%|

- IMGT: Better precision, more comprehensive reference
- NCBI: Higher recall, more permissive matching
- Both confirm same error patterns (IGHV ↔ IGKV confusion)

### Interpretation
- Model suitable for phylogenetic analysis at family level (IG/TR)
- Precision of 71.8% (IMGT) acceptable for evolutionary studies
- Lower recall (10-27%) due to:
  - Conservative TBLASTN e-value (1e-5)
  - Strain differences (training data vs GRCm39)
  - Comprehensive reference databases (more genes to find)
- Future improvement: consider 3-class model (background, IG, TR) to avoid intra-family confusion

### Files Structure
```
results/mouse/
├── VALIDATION_SUMMARY.md           # Comprehensive validation report
├── validation_imgt/
│   ├── validated.csv               # 79 correct predictions
│   ├── misclassified.csv           # 31 locus confusions
│   └── no_match.csv                # 92 without IMGT match
├── validation_ncbi/                # Strict validation (95% identity)
├── validation_ncbi_relaxed/        # Relaxed validation (80% identity)
└── phylogenetic_validation/
    ├── tree.nexus                  # For SeaView visualization
    └── tree.newick                 # Standard phylogenetic format
```

### Technical Notes
- IMGT references manually downloaded from web interface
- NCBI references extracted from GRCm39 GFF annotations
- ClustalO parameters: default settings, 8 threads
- BLAST databases created with makeblastdb
- Phylogenetic trees built with ClustalO guide tree method

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
