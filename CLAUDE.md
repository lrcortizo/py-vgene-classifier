# Contexto del investigador

Soy Luis Miguel Raña Cortizo, estudiante de doctorado a tiempo parcial en el
Doctorado en Sistemas Informáticos Inteligentes y Adaptativos (Universidade de
Vigo, Campus Ourense). Director: David Nicholas Olivieri Cecchi. Duración
prevista: 5 años (inicio 2025/2026).

## Proyecto de tesis
Título: "Phylo-Graph AI for Immune Repertoire Expansion and Contraction Across
Jawed Vertebrates"

Pregunta central: ¿Por qué algunas especies vertebradas tienen números
dramáticamente distintos de V-genes del receptor inmune, y está esa evolución
acoplada a otras vías biológicas (reparación de ADN, estrés metabólico)?

## Stack tecnológico
- Python, PyTorch, PyTorch Geometric
- BLAST+, ClustalO, MAFFT, OrthoFinder
- IMGT reference databases, NCBI genome assemblies (GCF accessions)
- Hardware: MSI Cyborg 15 A13V (RTX 4060, 8GB VRAM) + Lenovo ThinkPad P15
- Google Colab Pro para workloads GPU intensivos

## Preferencias de trabajo
- Proceder paso a paso, un comando a la vez
- Pegar output del terminal antes de continuar
- Documentar limitaciones honestamente, no sobrevendir resultados
- Mantener control de versiones Git cuidadoso
- Estilo conciso y directo, sin elaboración innecesaria
- Idioma de trabajo: español; código y comentarios técnicos en inglés

## Hitos administrativos pendientes
- Plan de investigación: 6 meses desde matriculación (requiere aprobación Olivieri)
- 450 horas de actividades formativas requeridas

---

# py-vgene-classifier — Instrucciones de repositorio

## Estructura del proyecto
scripts/           # Pipeline numerado 01-09 (ejecutar en orden)
src/
features/        # terminal_encoding.py — TerminalRegionEncoder (2000 dims)
models/          # cnn_terminal.py — CNN_TerminalEncoding (8.4M params)
utils/
parse_imgt_tables.py    # Species-agnostic (auto-detecta códigos de especie)
explore_vgenes.py
data/
raw/positive/          # V-genes IMGT por especie
background/            # Hard negatives (MHC, C-regions, Ig-superfamily)
processed/             # CSVs de entrenamiento/validación
genomes/               # Ensamblajes NCBI (.fna)
models/v2_*/             # Pesos entrenados (.pt)
results/{species}_identity60/   # Resultados por especie
## Clases y etiquetas
- 0 = background
- 1 = IGHV
- 2 = IGKV
- 3 = TRAV
- 4 = TRBV
- IGLV excluido deliberadamente — no es un error.

## Estado actual (abril 2026)
- Validación completada: ratón (GRCh38), humano, hurón
- IGHV/IGKV: buen rendimiento cross-species
- TRBV: fallo catastrófico cross-species (recall ~5% en hurón)
- Causa raíz confirmada: entrenamiento mono-especie (solo ratón)
- Tres intentos de remediación fallidos: rebalanceo, class weights, modelo híbrido
- SIGUIENTE PASO: entrenamiento multi-especie (~540 V-genes: ratón + humano
  + hurón)

## Reglas críticas al tocar el código
- Ratio recomendado V-gene:background = 1:3 (Olivieri). NO volver a
  ratios extremos (~175:1 causó colapso de TRBV).
- Deduplicación SIEMPRE por (query_id, sequence), nunca solo por sequence.
  Dedup por sequence sola destruye recall en clusters densos.
- Terminal cleaning activado por defecto (--clean-terminals).
- Threshold default = 0.5, configurable según use case.
- parse_imgt_tables.py es species-agnostic — no hardcodear especies.

## Workflow habitual
Ejecutar un script → pegar output del terminal → confirmar antes de continuar.
Nunca asumir que el paso anterior funcionó sin ver el output.

## Criterios de validación
- Recall calculado sobre UNIQUE IMGT genes encontrados (no total predicciones)
- BLASTP validation: min-identity 80%, min-coverage 70%
- Referencia de rendimiento actual (ratón): 93% recall, 99.8% precision

## Hitos de pipeline (v3.0.0)
- RSS-CAC boundary correction (v3.0.0): improves precision +2.9pp
  in Pongo pygmaeus by correcting C-terminal using CAC heptamer.
  1,646/3,691 candidates corrected in first test.

## Hardware disponible
- RTX 4060 (8GB VRAM): entrenamiento principal
- Si OOM: reducir batch-size a 32 antes de cualquier otra cosa
- Google Colab Pro: disponible para workloads que excedan VRAM local

## Convenciones de código
- Formatter: Black
- Linter: Flake8
- Type hints: Pylance
- Comentarios del código: inglés
- Commits: descriptivos, un cambio lógico por commit
