# Bootstrapping Protocol

## Concepto
Proceso iterativo inspirado en Olivieri & Gambón-Deza (2019).
En lugar de usar predicciones del pipeline sobre genomas nuevos,
se usan directamente secuencias funcionales de IMGT como semilla
para clados no representados en el training original.

## Iteraciones

### Bootstrap 1 (train_bootstrap.csv — 60,641 seqs)
- Fuente: predicciones validadas del pipeline sobre ratón, Pongo y hurón
- Umbral de inclusión: probabilidad ≥ 0.90 + validación BLASTP correcta
- Secuencias añadidas: +507 (mamíferos)
- Criterio de selección: solo predicciones con locus confirmado por BLAST

### Bootstrap 2 (train_bootstrap2.csv — 60,689 seqs)
- Fuente: secuencias funcionales IMGT de Xenopus laevis
- Secuencias añadidas: +48 (37 IGHV + 11 TRBV)
- Criterio de exclusión: secuencias con caracteres no estándar (X) o truncadas
- Excluidos: TRBV7*01 (X en posición central), TRBV10*01 (truncado en pos 1)
- Umbral de longitud: 80-140 aa

### Bootstrap 3 (train_bootstrap3.csv — 60,883 seqs)
- Fuente: secuencias funcionales IMGT de teleósteos
- Secuencias añadidas: +194
  - 35 TRBV Oncorhynchus mykiss (trucha arcoíris)
  - 132 TRAV Danio rerio (pez cebra)
  - 18 TRAV Takifugu rubripes (fugu)
  - 9 TRAV Oncorhynchus mykiss
- Criterio de deduplicación: se retiene un alelo por secuencia proteica idéntica
- Criterio de exclusión: caracteres no estándar, longitud fuera de rango
- Pares deduplicados: TRBV2S16≡S17, TRBV2S19≡S20, TRBV8S1≡S2

## Criterios generales de inclusión
- Longitud: 80-140 aa
- Sin caracteres no estándar (X, B, Z, etc.)
- Sin codones stop internos
- Para secuencias IMGT: solo entradas funcionales (F)
- Para predicciones del pipeline: probabilidad ≥ 0.90 + locus confirmado por BLAST

## Limitaciones conocidas
- RSS-CAC y Encoding V3 se introdujeron simultáneamente — no separables en ablación
- Xenopus TRBV (bootstrap2) causó caída de precisión TRBV en ratón (15.5%)
  corregida en bootstrap3 con TRBV de teleósteos
- No se realizó validación hold-out estricta entre iteraciones
