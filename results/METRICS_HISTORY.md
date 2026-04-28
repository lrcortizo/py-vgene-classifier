# V-Gene Classifier — Metrics History

> **Methodology**: Recall = unique IMGT genes found / Total IMGT functional genes
> (denominador = entradas funcionales en `all_vgenes_imgt.fasta`, numerador = `query=` IDs únicos en vgenes validados — metodología exacta de `scripts/08_validate_predictions.py`).
> Precision = predicciones correctas / total predicciones (nivel fila, por locus).

---

## Validation Results

| Especie | En training | Modelo | IGHV R | IGHV P | IGKV R | IGKV P | TRAV R | TRAV P | TRBV R | TRBV P |
|---------|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Mus musculus | Sí | v2.2 | 94.4% | 96.8% | 99.0% | 92.6% | 92.7% | 100.0% | 95.5% | 100.0% |
| Mus musculus | Sí | v3_bootstrap3 | 95.3% | 93.4% | 99.0% | 86.5% | 97.2% | 95.5% | 95.5% | 100.0% |
| Homo sapiens | Sí | v2.2 | 96.1% | 95.3% | 97.6% | 92.5% | 68.9% | 98.6% | 85.4% | 97.4% |
| Homo sapiens | Sí | v3_bootstrap3 | 96.1% | 91.0% | 97.6% | 92.4% | 88.9% | 84.0% | 85.4% | 94.1% |
| Pongo pygmaeus | No | v2.2 | 96.6% | 98.3% | 97.1% | 93.9% | 92.5% | 92.8% | 91.5% | 98.4% |
| Pongo pygmaeus | No | v3_bootstrap3 | 100.0% | 94.0% | 100.0% | 91.9% | 95.0% | 66.7% | 93.6% | 97.9% |
| Mustela p. furo | Sí | v2.2 | 92.9% | 84.4% | 90.0% | 99.5% | 76.9% | 97.0% | 85.0% | 100.0% |
| Mustela p. furo | Sí | v3_bootstrap3 | 95.2% | 82.7% | 92.5% | 99.8% | 88.5% | 94.7% | 85.0% | 100.0% |
| Xenopus laevis | No (parcial) | v2.2 | 65.8% | 97.6% | N/A | N/A | N/A | N/A | 8.3% | 100.0% |
| Xenopus laevis | No (parcial) | v3_bootstrap3 | 71.1% | 100.0% | N/A | N/A | N/A | N/A | 16.7% | 42.9% |

---

## Delta v2.2 → v3_bootstrap3

| Especie | ΔIGHV R | ΔIGKV R | ΔTRAV R | ΔTRBV R |
|---------|---------|---------|---------|---------|
| Mus musculus | +0.9pp | 0.0pp | +4.5pp | 0.0pp |
| Homo sapiens | 0.0pp | 0.0pp | +20.0pp | 0.0pp |
| Pongo pygmaeus | +3.4pp | +2.9pp | +2.5pp | +2.1pp |
| Mustela p. furo | +2.3pp | +2.5pp | +11.6pp | 0.0pp |
| Xenopus laevis | +5.3pp | N/A | N/A | +8.4pp |

---

## Notes

### v2.2 (baseline)
- Model: `v2_multispecies_r3` weights with multispecies reference panel
- Training set: bootstrap2 (~60,689 sequences, 7 mammal species including mouse, human, ferret)
- `--require-fr1` default: `True` for v3 encoder, `False` for v2

### v3_bootstrap3
- Model: `models/v3_bootstrap3/` (v3 encoder, trained on bootstrap3)
- Training set: bootstrap3 (60,883 sequences = bootstrap2 + 194 teleost sequences)
  - Added: 35 TRBV *Oncorhynchus mykiss*, 132 TRAV *Danio rerio*,
           18 TRAV *Takifugu rubripes*, 9 TRAV *Oncorhynchus mykiss*
- `--require-fr1` default changed to `False` for all encoder versions (commit `147400e`)
- Key wins: Human TRAV +20.0pp, Ferret TRAV +11.6pp, Mouse TRAV +4.5pp,
            Ferret IGHV/IGKV +2.3/+2.5pp, Pongo TRAV/TRBV +2.5/+2.1pp

### Xenopus laevis — TRBV precision drop (v3b3)
- TRBV precision drops from 100% to 42.9% in v3_bootstrap3
- Cause: teleost TRBV sequences added to training set share structural similarities with
  amphibian TRBV, causing more non-TRBV candidates to be classified as TRBV
- "No (parcial)": IGHV and TRBV sequences are included in the IMGT validation reference panel,
  but *X. laevis* is not part of the training set; IGKV/TRAV have no IMGT reference (N/A)

### Pongo pygmaeus — TRAV precision drop (v3b3)
- TRAV precision drops from 88.5% to 66.7%: the v3b3 model recovers 1 additional TRAV gene
  (38/40 vs 37/40) but produces proportionally more false-positive predictions

---

## Canonical validation runs used

| Especie | Modelo | Validation dir | vgenes FASTA |
|---------|--------|----------------|--------------|
| mouse | v2.2 | `mouse_v2.2/validation_rss` | `vgenes_rss.fasta` |
| mouse | v3b3 | `mouse_v2.2/validation_v3bootstrap3_nofr1` | `vgenes_v3bootstrap3_nofr1.fasta` |
| human | v2.2 | `human_v2.2/validation` | `vgenes_predicted.fasta` |
| human | v3b3 | `human_v2.2/validation_v3bootstrap3` | `vgenes_v3bootstrap3.fasta` |
| ferret | v2.2 | `ferret_v2.2/validation` | `vgenes_predicted.fasta` |
| ferret | v3b3 | `ferret_identity60/validation_v3bootstrap3_nofr1` | `vgenes_v3bootstrap3_nofr1.fasta` |
| pongo | v2.2 | `pongo_pygmaeus/validation` | `vgenes_predicted.fasta` |
| pongo | v3b3 | `pongo_pygmaeus/validation_v3bootstrap3_nofr1` | `vgenes_v3bootstrap3_nofr1.fasta` |
| xenopus | v2.2 | `xenopus_laevis/validation` | `vgenes_predicted.fasta` |
| xenopus | v3b3 | `xenopus_laevis/validation_v3bootstrap3` | `vgenes_v3bootstrap3.fasta` |

---

## Reference Counts (denominators used for recall)

| Locus | Mus musculus | Homo sapiens | Pongo pygmaeus | Mustela p. furo | Xenopus laevis |
|-------|-------------|-------------|----------------|-----------------|----------------|
| IGHV | 341 | 51 | 59 | 42 | 38 |
| IGKV | 100 | 42 | 35 | 40 | N/A |
| TRAV | 109 | 45 | 40 | 52 | N/A |
| TRBV | 22 | 48 | 47 | 20 | 12 |

*Counts from `all_vgenes_imgt.fasta` per species — `data/reference/imgt_<species>/`.*
