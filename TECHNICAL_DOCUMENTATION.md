# V-Gene Classifier — Technical Documentation

Pipeline version: 3.0.0  
Last updated: April 24, 2026

---

## Table of Contents

1. [Encoding Methods](#encoding-methods)
2. [Model Architecture](#model-architecture)
3. [Pipeline Scripts](#pipeline-scripts)
4. [Training Data](#training-data)
5. [Validation Protocol](#validation-protocol)
6. [v3.0.0 Changes](#v300-changes)

---

## Encoding Methods

### TerminalRegionEncoder (v2, 2000 dims)

Implements Olivieri's original encoding method for V-gene classification.

**Feature vector layout:**

| Slice | Component | Dims | Method |
|---|---|---|---|
| [0:800] | One-hot N-terminal 40 aa | 800 | Fixed-position one-hot over 20 AA |
| [800:1600] | One-hot C-terminal 40 aa | 800 | Fixed-position one-hot over 20 AA |
| [1600:2000] | Dipeptide counts | 400 | Integer counts of all 400 AA pairs |
| **Total** | | **2000** | |

**Rationale:**
- N-terminal 40 aa contains FR1 — the most discriminative region between loci
- C-terminal 40 aa contains FR3 with conserved motifs (YYC, C104, WGXG)
- Dipeptides capture global composition without length dependency

**Class:** `src.features.terminal_encoding.TerminalRegionEncoder`

---

### TerminalRegionEncoderV3 (v3, 2045 dims)

Extends TerminalRegionEncoder with 45 additional biologically-motivated features.

**Feature vector layout:**

| Slice | Component | Dims | Method |
|---|---|---|---|
| [0:1600] | One-hot terminals (N+C, 40aa each) | 1600 | Inherited from v2 |
| [1600:2000] | Dipeptide counts | 400 | Inherited from v2 |
| [2000:2020] | AA frequency histogram | 20 | Normalized marginal frequency over full sequence |
| [2020:2040] | Physicochemical properties | 20 | 5 scales x (mean, std) x (N-term, C-term) |
| [2040:2045] | Conserved motif flags | 5 | Binary flags, relative search windows |
| **Total** | | **2045** | |

**AA frequency histogram (20 dims):**
Normalized frequency of each of the 20 standard amino acids over the full sequence.
Complements dipeptides: dipeptides capture co-occurrences P(aa_i, aa_j); frequencies
capture marginals P(aa_i). Mathematically independent features.

**Physicochemical properties (20 dims):**
5 AAindex1 scales, each summarized as (mean, std) over N-terminal 40 aa and
C-terminal 40 aa separately: 5 x 2 x 2 = 20 dims.

| Property | Scale | Reference |
|---|---|---|
| Hydrophobicity | Kyte-Doolittle | Kyte & Doolittle 1982 |
| Molecular volume | Angstrom^3 | Pontius 1996 |
| Net charge | Approx. at pH 7 | Side chain pKa |
| Flexibility | Normalized B-factor | Vihinen 1994 |
| Polarity | Zimmerman scale | Zimmerman 1968 |

**Conserved motif flags (5 dims):**
Binary features using relative search windows (robust to FR1 length variation):

| Flag | Motif | Window | Biological meaning |
|---|---|---|---|
| flags[0] | C (Cys) | sequence[:30] | C23 proxy, FR1 conserved Cys |
| flags[1] | W (Trp) | sequence[35:45] | W41 proxy, FR2 conserved Trp |
| flags[2] | YYC | sequence[-25:] | FR3 conserved triad |
| flags[3] | C (Cys) | sequence[-20:] | C104 proxy, FR3 conserved Cys |
| flags[4] | WG.G (regex) | sequence[-25:] | WGXG motif, FR3-J junction |

**Class:** `src.features.terminal_encoding.TerminalRegionEncoderV3`

---

## Model Architecture

### CNN_TerminalEncoding

1D CNN treating the feature vector as a fixed-length sequence.

```
Input: (batch, input_size)          # 2000 (v2) or 2045 (v3)
  -> unsqueeze(1)
  -> (batch, 1, input_size)

Conv1: Conv1d(1, 64, k=5, pad=2) + BatchNorm1d(64) + ReLU + MaxPool1d(2)
  -> (batch, 64, input_size//2)

Conv2: Conv1d(64, 128, k=5, pad=2) + BatchNorm1d(128) + ReLU + MaxPool1d(2)
  -> (batch, 128, input_size//4)

Conv3: Conv1d(128, 256, k=5, pad=2) + BatchNorm1d(256) + ReLU + MaxPool1d(2)
  -> (batch, 256, input_size//8)

Flatten: (batch, 256 * (input_size//8))

FC: Linear(flatten, 128) + ReLU + Dropout(0.3)
    Linear(128, 64) + ReLU + Dropout(0.3)
    Linear(64, num_classes)

Output: (batch, 5) logits -> softmax -> class probabilities
```

**Flatten size (dynamic, computed from input_size):**

| input_size | After 3x MaxPool1d(2) | flatten_size | Parameters |
|---|---|---|---|
| 2000 | 250 | 64,000 | ~8.4M |
| 2045 | 255 | 65,280 | ~8.6M |

`flatten_size` is computed dynamically in `__init__` via:
```python
_s = input_size
for _ in range(3):
    _s = _s // 2
flatten_size = 256 * _s
```

**Class:** `src.models.cnn_terminal.CNN_TerminalEncoding`

---

## Pipeline Scripts

| Script | Purpose | Key parameters |
|---|---|---|
| 01 | Generate hard negative background sequences | --sources, --target-total |
| 01b | Expand background via conservative mutations | --mutation-rate |
| 02 | Prepare train/val CSV datasets | --background-ratio (recommended: 3.0) |
| 03 | Train CNN model | --encoder-version v2/v3, --epochs, --lr |
| 04 | Download genome assembly from NCBI | --accession, --method auto/wget |
| 05 | Run TBLASTN search against genome | --evalue, --threads |
| 06 | Extract and clean candidate sequences | --extend, --clean-terminals |
| 07 | Classify candidates with trained CNN | --encoder-version v2/v3, --threshold |
| 08 | Validate predictions against IMGT reference | --min-identity, --min-coverage |
| 09 | Optional phylogenetic tree validation | --top-n, --threads |

---

## Training Data

**Active training corpus (v2.2):** `data/processed_v2.2/train_multispecies_multiclass.csv`

| Class | Label | Sequences | Species |
|---|---|---|---|
| background | 0 | 45,100 | N/A (hard negatives) |
| IGHV | 1 | 4,584 | ~130 vertebrate species |
| IGKV | 2 | 3,207 | ~130 vertebrate species |
| TRAV | 3 | 4,457 | ~130 vertebrate species |
| TRBV | 4 | 2,786 | 130 species (12 from Mus musculus) |
| **Total** | | **60,134** | |

**Background:V-gene ratio:** 3:1 (45,100 background vs 15,034 V-genes)

**Known imbalance:** TRBV is underrepresented relative to other V-gene classes (4.6%
vs 7.6% for IGHV). Mouse TRBV has only 12 training sequences (0.43% of TRBV class),
which caused the V3 precision collapse on mouse TRBV before FR1 pattern fixes.

---

## Validation Protocol

**BLASTP validation (script 08):**
- Min identity: 80%
- Min coverage: 70%
- Metric: unique IMGT genes found (not total prediction count)

**Recall definition:**
`recall = len(unique IMGT genes with BLAST match) / len(total IMGT reference genes)`

**Precision definition:**
`precision = len(correct locus predictions) / len(total predictions above threshold)`

**Validated species:**

| Species | Genome | In training? | Model used |
|---|---|---|---|
| Mus musculus | GCF_000001635.27 | Yes | v2_multispecies_r3 / v3_multispecies |
| Homo sapiens | GRCh38 | Yes | v2_multispecies_r3 |
| Mustela putorius furo | GCF_011764305.1 | Yes | v2_multispecies_r3 |
| Pongo pygmaeus | GCF_028885625.2 | No | v2_multispecies_r3 / v3_multispecies |

---

## v3.0.0 Changes

### 1. RSS-CAC C-Terminal Boundary Correction

**Location:** `scripts/06_extract_candidates.py`, function `extract_sequences()`

**Problem:** The extracted ORF may extend past the true V-gene C-terminus into the
recombination signal sequence (RSS) and beyond if no stop codon appears immediately
after the last coding codon. `clean_terminals` uses `max_vgene_length=120` as a
hard cutoff but does not detect the exact exon boundary.

**Solution:** After selecting `best_protein`, search for the RSS CAC heptamer within
±15 nt of the predicted C-terminus. If CAC is found closer to the ORF start:

```python
protein_end_in_dna = best_frame_start_in_dna + len(best_protein) * 3
search_start = max(0, protein_end_in_dna - 15)
window = str(dna_seq[search_start : protein_end_in_dna + 15])
cac_offset = window.find('CAC')
if cac_offset != -1:
    cac_pos_in_dna = search_start + cac_offset
    corrected_len = (cac_pos_in_dna - best_frame_start_in_dna) // 3
    if 70 <= corrected_len < len(best_protein):
        best_protein = best_protein[:corrected_len]
        rss_corrected_count += 1
```

**Key design decisions:**
- `best_frame_start_in_dna` is tracked for both preferred-frame and fallback paths
- Window is ±15 nt (not just downstream) to catch proteins that are slightly too long
- Minimum corrected length: 70 aa (same as general extraction minimum)
- Conservative: no change if CAC not found

**Validation result (Pongo pygmaeus):**
- 1,646/3,691 candidates corrected (44.6%)
- Precision: 97.0% → 99.9% (+2.9 pp)
- Recall: unchanged (96.1%)

**Reference:** Olivieri 2019 — CAC conserved in all jawed vertebrates (<1% variation)

---

### 2. TerminalRegionEncoderV3 (2045 dims)

See [Encoding Methods](#encoding-methods) section above for full technical description.

**Implementation notes:**
- Inherits `_onehot_extremes()` and `_dipeptide_counts()` from `TerminalRegionEncoder`
- Only overrides `encode()` and sets `self.n_features = 2045`
- `PHYSICOCHEMICAL` dict and `_PROP_NAMES` list defined at module level
- `re` module used for WGXG motif detection

**Backward compatibility:**
- `TerminalRegionEncoder` (2000 dims) unchanged
- All existing models continue to work without modification
- V3 requires explicit `--encoder-version v3` flag in scripts 03 and 07

---

### 3. FR1 Pattern Expansion (43 → 47 patterns)

**Location:** `scripts/06_extract_candidates.py`, `VGENE_START_PATTERNS`

Four new patterns added during v3.0.0 development:

| Pattern | Locus | Covers | FP in other loci |
|---|---|---|---|
| `QVTL` | IGHV | VH2 family (Pongo IGHV2-48, IGHV2-132) | 0 |
| `ETT[LV]TQ` | IGKV | ETT family (Pongo IGKV5-2, mouse IGKV) | 0 |
| `QAYL` | IGHV | VH4-like family (mouse IGHV1-12) | 0 |
| `QREL` | IGHV | VH4-like family (mouse IGHV1-49) | 0 |

**QAYL/QREL context:** Without these patterns, IGHV1-12 and IGHV1-49 candidates
had a 16 aa N-terminal junk prefix (`DCTDIHSAFPSIGVHS`). The V3 encoder's
physicochemical features computed over this junk prefix caused 415 IGHV candidates
to be misclassified as TRBV with prob_TRBV median=1.0.

---

### 4. v2 vs v3 Validation Comparison

#### Mouse (GRCm39, Mus musculus)

| Locus | v2_multispecies_r3 | v3_multispecies (v3fix) | Delta |
|---|---|---|---|
| IGHV recall | 94.4% | 95.3% | +0.9 pp |
| IGKV recall | 97.0% | 99.0% | +2.0 pp |
| TRAV recall | 92.7% | 92.7% | — |
| TRBV recall | 95.5% | 95.5% | — |
| Overall recall | 94.6% | 95.6% | +1.0 pp |
| Precision | 95.4% | 97.4% | +2.0 pp |

#### Pongo pygmaeus (mPonPyg2, out-of-training)

| Locus | v2_multispecies_r3 | v3_multispecies | Delta |
|---|---|---|---|
| IGHV recall | 100.0% | 100.0% | — |
| IGKV recall | 100.0% | 100.0% | — |
| TRAV recall | 92.5% | 94.6% | +2.1 pp |
| TRBV recall | 91.5% | 93.6% | +2.1 pp |
| Overall recall | 96.1% | 96.1% | — |
| Precision | 97.0% | 99.9% | +2.9 pp |

**Known remaining issue:** 185 IGKV candidates classified as IGHV at 88.4% BLAST
identity. Under investigation — likely IGKV/IGHV boundary cases in training data.

---

## Known Limitations

### IGHV1-26*01 (mouse) — Truncated FR1 due to insufficient extension

**Symptom:** 185 IGKV predictions in mouse v3fix validation that BLAST back to IGHV
at 88.4% identity.

**Root cause:** All 185 are duplicate predictions of the same single sequence.
TBLASTN anchors approximately 8 aa into the gene body (skipping the first 8 aa of FR1:
`EVQLQQSG`). The `--extend 150` window is insufficient to recover the true V-gene start
for this locus. The extracted sequence begins with `AGEPGASVK`, a non-canonical
N-terminal that does not match any VGENE_START_PATTERNS.

**Classifier behavior:** The model sees an unusual N-terminal and correctly identifies
it as not fitting the canonical IGHV profile. It classifies as IGKV with prob=0.85.
This is internally consistent behavior — the error is in extraction, not classification.

**Impact:** 1 unique gene lost (IGHV1-26*01), not 185 independent errors.
The gene appears 185 times because multiple TBLASTN hits anchor to the same locus from
different query sequences.

**Fix (not yet implemented):** Increase `--extend` to 200 nt, or implement adaptive
extension when no FR1 pattern is found in the extracted candidate (re-extract with
larger window before discarding).

---

*End of TECHNICAL_DOCUMENTATION.md*
