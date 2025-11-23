"""
Run TBLASTN to find V-gene candidates in target genome
"""
import subprocess
from pathlib import Path
import pandas as pd
import sys

def create_blast_db(genome_fasta, db_name):
    """Create BLAST database from genome"""
    print(f"\n{'='*70}")
    print(f"CREATING BLAST DATABASE")
    print(f"{'='*70}")
    print(f"Genome: {genome_fasta}")
    print(f"Database: {db_name}")
    
    cmd = [
        "makeblastdb",
        "-in", str(genome_fasta),
        "-dbtype", "nucl",
        "-out", str(db_name)
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✅ BLAST database created")

def run_tblastn(query_fasta, db_name, output_file, evalue=1e-5):
    """
    Run TBLASTN search
    
    Args:
        query_fasta: FASTA with V-gene proteins (query)
        db_name: BLAST database name
        output_file: Output file for hits
        evalue: E-value threshold
    """
    print(f"\n{'='*70}")
    print(f"RUNNING TBLASTN")
    print(f"{'='*70}")
    print(f"Query: {query_fasta}")
    print(f"Database: {db_name}")
    print(f"E-value threshold: {evalue}")
    
    cmd = [
        "tblastn",
        "-query", str(query_fasta),
        "-db", str(db_name),
        "-out", str(output_file),
        "-outfmt", "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore sframe",
        "-evalue", str(evalue),
        "-max_target_seqs", "10000",
        "-num_threads", "4"
    ]
    
    print(f"\n🔍 Searching... (this may take 10-30 minutes)")
    subprocess.run(cmd, check=True)
    print(f"\n✅ TBLASTN complete")
    
    # Parse and show summary
    df = pd.read_csv(
        output_file,
        sep="\t",
        names=["qseqid", "sseqid", "pident", "length", "qstart", "qend", 
               "sstart", "send", "evalue", "bitscore", "sframe"]
    )
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total hits: {len(df)}")
    print(f"Unique query V-genes matched: {df['qseqid'].nunique()}")
    print(f"Unique genomic contigs hit: {df['sseqid'].nunique()}")
    print(f"Mean % identity: {df['pident'].mean():.1f}%")
    print(f"Max % identity: {df['pident'].max():.1f}%")
    print(f"Min % identity: {df['pident'].min():.1f}%")
    
    # Show top hits
    print(f"\nTop 10 hits by bitscore:")
    print(df.nlargest(10, 'bitscore')[['qseqid', 'sseqid', 'pident', 'evalue', 'bitscore']].to_string())
    
    return df

if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).parent.parent
    DATA_DIR = PROJECT_DIR / "data"
    RESULTS_DIR = PROJECT_DIR / "results"
    
    # Species (configurable)
    SPECIES = "myotis_lucifugus"
    
    # Create species results directory
    SPECIES_RESULTS = RESULTS_DIR / SPECIES
    SPECIES_RESULTS.mkdir(parents=True, exist_ok=True)
    
    # Paths
    GENOME_DIR = DATA_DIR / "genomes" / SPECIES
    
    # Find genome FASTA (searches recursively)
    genome_files = list(GENOME_DIR.rglob("*.fna"))
    if not genome_files:
        print(f"❌ No genome FASTA (.fna) found in {GENOME_DIR}")
        print(f"   Make sure you ran 05_download_genome.py first")
        sys.exit(1)
    
    GENOME_FASTA = genome_files[0]
    print(f"Found genome: {GENOME_FASTA}")
    
    # Query V-genes (human IGHV for now)
    QUERY_FASTA = DATA_DIR / "raw" / "positive" / "ighv.fasta"
    
    if not QUERY_FASTA.exists():
        print(f"❌ Query file not found: {QUERY_FASTA}")
        sys.exit(1)
    
    # Output paths
    DB_NAME = GENOME_DIR / f"{SPECIES}_genome_db"
    OUTPUT_FILE = SPECIES_RESULTS / "tblastn_hits.txt"
    
    # 1. Create BLAST database (if not exists)
    if not Path(f"{DB_NAME}.nhr").exists():
        create_blast_db(GENOME_FASTA, DB_NAME)
    else:
        print(f"\n✅ BLAST database already exists: {DB_NAME}")
    
    # 2. Run TBLASTN
    hits_df = run_tblastn(QUERY_FASTA, DB_NAME, OUTPUT_FILE)
    
    print(f"\n{'='*70}")
    print(f"✅ DONE")
    print(f"{'='*70}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"\nNext step: python scripts/07_extract_candidates.py")