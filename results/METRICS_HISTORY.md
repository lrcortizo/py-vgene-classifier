# V-Gene Classifier — Metrics History

> **Methodology**: Recall = unique IMGT genes found / Total IMGT functional genes
> (denominador = entradas funcionales en `all_vgenes_imgt.fasta`, numerador = `query=` IDs únicos en vgenes validados — metodología exacta de `scripts/08_validate_predictions.py`).
> Precision = predicciones correctas / total predicciones (nivel fila, por locus).

---

## Validation Results

| Especie | En training | Modelo | IGHV R | IGHV P | IGKV R | IGKV P | TRAV R | TRAV P | TRBV R | TRBV P |
|---------|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Mus musculus | No | v2.2 | 94.4% | 96.8% | 97.0% | 92.6% | 92.7% | 100.0% | 59.1% | 100.0% |
| Mus musculus | No | v3_bootstrap3 | 95.3% | 97.0% | 99.0% | 96.1% | 92.7% | 98.5% | 95.5% | 100.0% |
| Homo sapiens | No | v2.2 | 96.1% | 95.3% | 97.6% | 92.5% | 68.9% | 98.6% | 85.4% | 97.4% |
| Homo sapiens | No | v3_bootstrap3 | 96.1% | 91.0% | 97.6% | 92.4% | 88.9% | 84.0% | 85.4% | 94.1% |
| Pongo pygmaeus | No | v2.2 | 96.6% | 98.3% | 97.1% | 93.9% | 92.5% | 92.8% | 91.5% | 98.4% |
| Pongo pygmaeus | No | v3_bootstrap3 | 100.0% | 97.8% | 100.0% | 93.9% | 90.0% | 85.0% | 91.5% | 97.9% |
| Mustela p. furo | Sí | v2.2 | 92.9% | 84.4% | 90.0% | 99.5% | 76.9% | 97.0% | 85.0% | 100.0% |
| Mustela p. furo | Sí | v3_bootstrap3 | 92.9% | 88.3% | 90.0% | 99.0% | 82.7% | 93.9% | 85.0% | 100.0% |
| Xenopus laevis | No | v2.2 | 65.8% | 97.6% | N/A | N/A | N/A | N/A | 8.3% | 100.0% |
| Xenopus laevis | No | v3_bootstrap3 | 71.1% | 100.0% | N/A | N/A | N/A | N/A | 16.7% | 42.9% |

---

## Delta v2.2 → v3_bootstrap3

| Especie | ΔIGHV R | ΔIGKV R | ΔTRAV R | ΔTRBV R |
|---------|---------|---------|---------|---------|
| Mus musculus | +0.9pp | +2.0pp | 0.0pp | +36.4pp |
| Homo sapiens | 0.0pp | 0.0pp | +20.0pp | 0.0pp |
| Pongo pygmaeus | +3.4pp | +2.9pp | -2.5pp | 0.0pp |
| Mustela p. furo | 0.0pp | 0.0pp | +5.8pp | 0.0pp |
| Xenopus laevis | +5.3pp | N/A | N/A | +8.4pp |

---

## Notes

### v2.2 (baseline)
- Model: `v2_multispecies_r3` weights with multispecies reference panel
- Training set: bootstrap2 (~60,689 sequences, 7 mammal species)
- `--require-fr1` default: `True` for v3 encoder, `False` for v2

### v3_bootstrap3
- Model: `models/v3_bootstrap3/` (v3 encoder, trained on bootstrap3)
- Training set: bootstrap3 (60,883 sequences = bootstrap2 + 194 teleost sequences)
  - Added: 35 TRBV *Oncorhynchus mykiss*, 132 TRAV *Danio rerio*,
           18 TRAV *Takifugu rubripes*, 9 TRAV *Oncorhynchus mykiss*
- `--require-fr1` default changed to `False` for all encoder versions (commit `147400e`)
- Key wins: Mouse TRBV +36.4pp, Human TRAV +20.0pp, Ferret TRAV +5.8pp, Pongo IGHV/IGKV +3–4pp

### Xenopus laevis — TRBV precision drop (v3b3)
- TRBV precision drops from 100% to 42.9% in v3_bootstrap3
- Cause: teleost TRBV sequences added to training set share structural similarities with amphibian TRBV,
  causing some non-TRBV candidates to be classified as TRBV
- IGKV/TRAV: no IMGT reference available for *X. laevis* (N/A in all versions)

---

## Reference Counts (denominators used for recall)

| Locus | Mus musculus | Homo sapiens | Pongo pygmaeus | Mustela p. furo | Xenopus laevis |
|-------|-------------|-------------|----------------|-----------------|----------------|
| IGHV | 107 | 77 | 29 | 28 | 38 |
| IGKV | 100 | 41 | 35 | 20 | N/A |
| TRAV | 96 | 45 | 40 | 52 | N/A |
| TRBV | 22 | 41 | 47 | 20 | 12 |

*Counts from `all_vgenes_imgt.fasta` per species validation run.*
