#!/usr/bin/env python3
"""
Focused phylogenetic analysis for the 10 Pongo pygmaeus genes absent from predictions.

Builds a small tree (~50-60 sequences) containing:
  - 10 missing IMGT reference genes (the ones not recovered)
  - 5 correctly classified IGHV genes (high-confidence anchors)
  - Misclassified candidates (from candidates.fasta, queried against the 10 missing genes)

Usage:
    python scripts/10_focused_phylo_pongo.py
"""

import subprocess
import sys
import csv
import shutil
from pathlib import Path

# Locate clustalo — works whether it's on PATH or only in the conda env Scripts/
CLUSTALO = shutil.which("clustalo") or shutil.which("clustalo.exe")
if CLUSTALO is None:
    sys.exit(
        "[ERR] clustalo not found. Install with:\n"
        "  conda install -n phd -c bioconda clustalo\n"
        "Or place clustalo.exe in your PATH."
    )
print(f"clustalo found at: {CLUSTALO}")

# -- Configuration --------------------------------------------------------------

MISSING_GENES = [
    "IGHV2-132", "IGHV2-48",
    "IGKV5-2",
    "TRAV1-1", "TRAV1-2", "TRAV30",
    "TRBV15", "TRBV2-1", "TRBV2-2", "TRBV2-4",
]

IMGT_REF      = Path("data/reference/imgt_pongo_pygmaeus/all_vgenes_imgt.fasta")
CANDIDATES    = Path("results/pongo_pygmaeus/candidates.fasta")
PRED_CSV      = Path("results/pongo_pygmaeus/vgenes_predicted_predictions.csv")
VALIDATED_CSV = Path("results/pongo_pygmaeus/validation/validated.csv")
OUTPUT_DIR    = Path("results/pongo_pygmaeus/phylogenetic_focused")

MAX_CANDIDATES_PER_GENE = 3   # cap per missing gene to keep tree readable
TOP_N_IGHV_ANCHORS      = 5   # correctly classified IGHV anchors


# -- Helpers --------------------------------------------------------------------

def parse_fasta(path):
    """Return list of (header, seq) tuples."""
    records = []
    header, seq_parts = None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def write_fasta(records, path):
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


def run_clustalo(input_fasta, output_aln, threads=8):
    print(f"\n  Running ClustalO ({input_fasta.name} -> {output_aln.name})...")
    cmd = [
        CLUSTALO,
        "--infile",  str(input_fasta),
        "--outfile", str(output_aln),
        "--outfmt",  "fasta",
        "--threads", str(threads),
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERR] ClustalO failed:\n{result.stderr}")
        sys.exit(1)
    print(f"  [OK] Alignment done")


def build_tree(input_aln, output_newick, output_nexus, threads=8):
    print(f"\n  Building tree ({input_aln.name})...")
    cmd = [
        CLUSTALO,
        "--infile",        str(input_aln),
        "--guidetree-out", str(output_newick),
        "--outfmt",        "fasta",
        "--outfile",       str(output_newick.with_suffix(".aln.fasta")),
        "--threads",       str(threads),
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERR] Tree failed:\n{result.stderr}")
        sys.exit(1)

    # Wrap newick -> nexus
    newick_str = output_newick.read_text().strip()
    nexus = (
        "#NEXUS\nbegin trees;\n"
        f"  tree 1 = {newick_str}\n"
        "end;\n"
    )
    output_nexus.write_text(nexus)
    print(f"  [OK] tree.newick -> {output_newick}")
    print(f"  [OK] tree.nexus  -> {output_nexus}")


# -- Main -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FOCUSED PHYLOGENETIC ANALYSIS -- Pongo pygmaeus missing genes")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. Extract missing genes from IMGT reference --------------------------
    print("\n[1] Extracting 10 missing genes from IMGT reference...")
    all_ref = parse_fasta(IMGT_REF)
    missing_ref = []
    for header, seq in all_ref:
        gene_field = next((f for f in header.split() if f.startswith("gene=")), "")
        gene_name  = gene_field.replace("gene=", "")
        if any(gene_name == mg or gene_name.startswith(mg + "*") for mg in MISSING_GENES):
            label = f"REF_{header.split()[0]}"
            missing_ref.append((label, seq))
            print(f"  [OK] {label}")

    print(f"  -> {len(missing_ref)} reference sequences selected")

    # -- 2. IGHV anchors: top-5 correctly validated IGHV by prob_IGHV ----------
    print(f"\n[2] Selecting {TOP_N_IGHV_ANCHORS} IGHV anchor sequences...")

    pred_by_id = {}
    with open(PRED_CSV) as f:
        for row in csv.DictReader(f):
            pred_by_id[row["id"]] = row

    validated_ighv = []
    with open(VALIDATED_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("ref_locus", "").upper() == "IGHV" and row["id"] in pred_by_id:
                prob = float(pred_by_id[row["id"]]["prob_IGHV"])
                validated_ighv.append((prob, row["id"], row["sequence"]))

    validated_ighv.sort(reverse=True)
    anchors = []
    for prob, cid, seq in validated_ighv[:TOP_N_IGHV_ANCHORS]:
        label = f"ANCHOR_{cid}_prob{prob:.4f}"
        anchors.append((label, seq))
        print(f"  [OK] {label}")

    print(f"  -> {len(anchors)} anchor sequences selected")

    # -- 3. Misclassified candidates for the 10 missing genes -----------------
    print(f"\n[3] Extracting misclassified candidates for the 10 missing genes...")
    all_cands = parse_fasta(CANDIDATES)

    misclassified = []
    seen_genes = {}

    for header, seq in all_cands:
        cid = header.split()[0]
        query_field = next((f for f in header.split() if f.startswith("query=")), "")
        query_gene  = query_field.replace("query=", "")
        gene_base   = query_gene.split("*")[0]

        if gene_base not in MISSING_GENES:
            continue

        seen_genes.setdefault(gene_base, 0)
        if seen_genes[gene_base] >= MAX_CANDIDATES_PER_GENE:
            continue

        pred = pred_by_id.get(cid, {})
        pred_locus = pred.get("predicted_locus", "unknown")
        prob_raw   = pred.get("probability", "0")
        prob       = float(prob_raw) if prob_raw else 0.0

        label = f"CAND_{cid}_query{gene_base}_pred{pred_locus}_p{prob:.2f}"
        misclassified.append((label, seq))
        seen_genes[gene_base] += 1
        print(f"  [OK] {label}")

    print(f"  -> {len(misclassified)} candidate sequences selected")

    # -- 4. Combine and write input FASTA --------------------------------------
    combined = missing_ref + anchors + misclassified
    print(f"\n[4] Total sequences for tree: {len(combined)}")
    combined_fasta = OUTPUT_DIR / "combined.fasta"
    write_fasta(combined, combined_fasta)
    print(f"  -> {combined_fasta}")

    # -- 5. Align with ClustalO ------------------------------------------------
    alignment_fasta = OUTPUT_DIR / "alignment.fasta"
    run_clustalo(combined_fasta, alignment_fasta)

    # -- 6. Build tree ---------------------------------------------------------
    tree_newick = OUTPUT_DIR / "tree.newick"
    tree_nexus  = OUTPUT_DIR / "tree.nexus"
    build_tree(alignment_fasta, tree_newick, tree_nexus)

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nSequences in tree: {len(combined)}")
    print(f"  REF_*    (missing IMGT genes):  {len(missing_ref)}")
    print(f"  ANCHOR_* (correct IGHV):        {len(anchors)}")
    print(f"  CAND_*   (misclassified):       {len(misclassified)}")
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("  combined.fasta   -- raw input")
    print("  alignment.fasta  -- MSA")
    print("  tree.newick      -- Newick tree")
    print("  tree.nexus       -- Nexus tree (for SeaView)")
    print(f"\nVisualize: seaview {tree_nexus}")


if __name__ == "__main__":
    main()
