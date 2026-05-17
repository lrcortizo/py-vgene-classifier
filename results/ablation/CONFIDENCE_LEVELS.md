# Confidence Level System

## Criterios finales (encoder v3)
- high:   prob >= 0.99 AND n_motifs >= 4 (de C23, W41, YYC, C104, WGXG)
- medium: prob >= 0.80 AND n_motifs >= 2
- low:    resto

## Rendimiento empírico (Mus musculus, v3_bootstrap3)
| Nivel  | N      | %     | Precisión |
|--------|--------|-------|-----------|
| high   | 7,031  | 40.5% | 98.1%     |
| medium | 10,016 | 57.7% | 89.1%     |
| low    | 314    | 1.8%  | 89.8%     |

## Sistemas evaluados y descartados
1. prob+margin: 97.8% en high — distribución degenerada, no informativa
2. n>=3 AND W41: high=98.9% pero medium=76.1% — degrada el tier medio

## Limitaciones conocidas
- TRBV nunca alcanza high: el triptófano W41 y otros motivos
  tienen posiciones ligeramente distintas en TRBV respecto a
  IGHV/IGKV/TRAV. Todo TRBV cae en medium (precisión 100%
  empíricamente en ratón).
- Los FPs en high son pseudogenes o V-genes no anotados en IMGT
  estructuralmente completos — no son distinguibles de TPs
  por motivos conservados.
- WGXG nunca aparece en los últimos 25aa de los candidatos
  de ratón — la ventana puede necesitar ajuste para otros taxa.
- Para encoder v2: sistema de fallback prob+margin (sin motivos).
