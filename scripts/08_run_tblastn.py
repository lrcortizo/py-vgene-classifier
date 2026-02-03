#!/usr/bin/env python3
"""
Run TBLASTN to find V-gene candidates in a genome.

This script uses human V-genes as queries to search for homologous
sequences in a target genome using TBLASTN.

Usage:
    python scripts/08_run_tblastn.py \
        --genome data/genomes/myotis_lucifugus/GCF_000147115.1.fna \
        --query data/raw/positive/human_vgenes.fasta \
        --output results/bat/tblastn_results.txt \
        --evalue 1e-5 \
        --threads 4
"""

import argparse
import subprocess
from pathlib import Path
import time


def run_makeblastdb(genome_fasta: Path, db_path: Path) -> bool:
    """
    Create BLAST database from genome.

    Returns True if successful, False otherwise.
    """
    print(f"\n🔨 Creating BLAST database...")
    print(f"   Input: {genome_fasta}")
    print(f"   Output: {db_path}")

    try:
        cmd = [
            "makeblastdb",
            "-in", str(genome_fasta),
            "-dbtype", "nucl",
            "-out", str(db_path),
            "-parse_seqids"
        ]

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print(f"   ✅ Database created")
        return True

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error creating database: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"   ❌ makeblastdb not found. Install BLAST+:")
        print(f"      sudo apt install ncbi-blast+")
        return False


def run_tblastn(
    query_fasta: Path,
    db_path: Path,
    output_file: Path,
    evalue: float = 1e-5,
    threads: int = 4
) -> bool:
    """
    Run TBLASTN search.

    Returns True if successful, False otherwise.
    """
    print(f"\n🔍 Running TBLASTN...")
    print(f"   Query: {query_fasta}")
    print(f"   Database: {db_path}")
    print(f"   E-value: {evalue}")
    print(f"   Threads: {threads}")

    start_time = time.time()

    try:
        cmd = [
            "tblastn",
            "-query", str(query_fasta),
            "-db", str(db_path),
            "-out", str(output_file),
            "-evalue", str(evalue),
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen",
            "-num_threads", str(threads),
            "-max_target_seqs", "50000"  # Increase to capture all hits
        ]

        print(f"\n   ⏳ Searching... (this may take 5-15 minutes)")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start_time

        print(f"   ✅ TBLASTN completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error running TBLASTN: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"   ❌ tblastn not found. Install BLAST+:")
        print(f"      sudo apt install ncbi-blast+")
        return False


def analyze_results(output_file: Path):
    """Analyze TBLASTN results."""
    print(f"\n📊 Analyzing results...")

    if not output_file.exists():
        print(f"   ❌ Results file not found")
        return

    # Count hits
    with open(output_file) as f:
        lines = [line for line in f if line.strip()]

    if not lines:
        print(f"   ⚠️  No hits found")
        return

    print(f"   Total hits: {len(lines):,}")

    # Parse results
    hits_by_query = {}
    for line in lines:
        fields = line.strip().split('\t')
        query = fields[0]
        hits_by_query[query] = hits_by_query.get(query, 0) + 1

    print(f"   Unique queries with hits: {len(hits_by_query)}")
    print(f"   Avg hits per query: {len(lines)/len(hits_by_query):.1f}")

    # Top queries
    top_queries = sorted(hits_by_query.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n   Top 5 queries:")
    for query, count in top_queries:
        print(f"      {query}: {count} hits")


def main():
    parser = argparse.ArgumentParser(
        description="Run TBLASTN to find V-gene candidates"
    )
    parser.add_argument(
        "--genome",
        type=Path,
        required=True,
        help="Genome FASTA file"
    )
    parser.add_argument(
        "--query",
        type=Path,
        required=True,
        help="Query sequences (V-genes) FASTA file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for TBLASTN results"
    )
    parser.add_argument(
        "--evalue",
        type=float,
        default=1e-5,
        help="E-value threshold (default: 1e-5)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads (default: 4)"
    )
    parser.add_argument(
        "--skip-makedb",
        action="store_true",
        help="Skip database creation (if already exists)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RUN TBLASTN V-GENE SEARCH")
    print("=" * 70)

    # Validate inputs
    if not args.genome.exists():
        print(f"\n❌ Genome file not found: {args.genome}")
        return

    if not args.query.exists():
        print(f"\n❌ Query file not found: {args.query}")
        return

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Database path (same directory as genome)
    db_path = args.genome.parent / args.genome.stem

    # Create BLAST database
    if not args.skip_makedb:
        if not run_makeblastdb(args.genome, db_path):
            return
    else:
        print(f"\n⏭️  Skipping database creation")
        print(f"   Using existing: {db_path}")

    # Run TBLASTN
    if not run_tblastn(args.query, db_path, args.output, args.evalue, args.threads):
        return

    # Analyze results
    analyze_results(args.output)

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"\nResults saved: {args.output}")
    print("\nNext step:")
    print(f"  python scripts/09_extract_candidates.py \\")
    print(f"      --tblastn-results {args.output} \\")
    print(f"      --genome {args.genome} \\")
    print(f"      --output results/mouse/candidates.fasta")


if __name__ == "__main__":
    main()
