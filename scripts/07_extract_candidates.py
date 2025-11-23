"""
Extract candidate V-gene sequences from BLAST hits
"""
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from pathlib import Path
import sys

def load_genome(genome_file):
    """Load genome into memory as dictionary"""
    print(f"\n📖 Loading genome: {genome_file}")
    print(f"   (this may take a minute for large genomes...)")
    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
    print(f"   ✅ Loaded {len(genome)} contigs/chromosomes")
    return genome

def extract_region(genome, contig_id, start, end, strand):
    """
    Extract a genomic region
    
    Args:
        genome: Dictionary of SeqIO records
        contig_id: Contig/chromosome ID
        start, end: Coordinates (1-indexed)
        strand: +1 or -1
    
    Returns:
        DNA sequence (Seq object)
    """
    if contig_id not in genome:
        return None
    
    # Extract region (convert to 0-indexed)
    seq = genome[contig_id].seq[start-1:end]
    
    # Reverse complement if negative strand
    if strand < 0:
        seq = seq.reverse_complement()
    
    return seq

def find_start_stop(dna_seq, min_length=250, max_length=450):
    """
    Find START (ATG) and STOP codons in sequence
    
    Args:
        dna_seq: DNA sequence
        min_length, max_length: Expected gene length in bp
    
    Returns:
        (start_pos, stop_pos) or None
    """
    # Find all ATG positions
    start_codons = []
    for i in range(len(dna_seq) - 2):
        if str(dna_seq[i:i+3]) == "ATG":
            start_codons.append(i)
    
    # Find all STOP positions
    stop_codons = []
    stops = ["TAA", "TAG", "TGA"]
    for i in range(len(dna_seq) - 2):
        if str(dna_seq[i:i+3]) in stops:
            stop_codons.append(i)
    
    # Find best START-STOP pair
    # Criterios: longitud ~300-400 bp, in-frame
    best_pair = None
    best_score = float('inf')
    
    for start in start_codons:
        for stop in stop_codons:
            if stop > start:
                length = stop - start
                # Check if in-frame (múltiplo de 3)
                if (stop - start) % 3 != 0:
                    continue
                # Check length
                if min_length <= length <= max_length:
                    # Score: prefer length close to 350
                    score = abs(length - 350)
                    if score < best_score:
                        best_pair = (start, stop)
                        best_score = score
    
    return best_pair

def extract_candidates(hits_file, genome_file, output_fasta, extend=500):
    """
    Extract candidate V-gene sequences from TBLASTN hits
    
    Args:
        hits_file: TBLASTN output (tabular)
        genome_file: Genome FASTA
        output_fasta: Output FASTA with candidates
        extend: Bases to extend around hit (to find START/STOP)
    """
    # Load genome
    genome = load_genome(genome_file)
    
    # Load hits
    print(f"\n📊 Loading TBLASTN hits...")
    hits_df = pd.read_csv(
        hits_file,
        sep="\t",
        names=["qseqid", "sseqid", "pident", "length", "qstart", "qend", 
               "sstart", "send", "evalue", "bitscore", "sframe"]
    )
    print(f"   Total hits: {len(hits_df)}")
    
    # Filter by quality (optional)
    # Keep only hits with good identity and e-value
    filtered = hits_df[
        (hits_df['pident'] >= 60) &  # At least 60% identity
        (hits_df['evalue'] <= 1e-5)   # Good e-value
    ]
    print(f"   After filtering (pident>=60%, evalue<=1e-5): {len(filtered)}")
    
    print(f"\n🔬 Extracting candidate sequences...")
    candidates = []
    
    for idx, hit in filtered.iterrows():
        # Progress
        if idx % 50 == 0:
            print(f"   Progress: {idx}/{len(filtered)}", end='\r')
        
        # Determine strand
        strand = 1 if hit['sstart'] < hit['send'] else -1
        start = min(hit['sstart'], hit['send']) - extend
        end = max(hit['sstart'], hit['send']) + extend
        
        # Ensure positive coordinates
        start = max(1, start)
        
        # Extract region
        region_seq = extract_region(
            genome, 
            hit['sseqid'], 
            start, 
            end, 
            strand
        )
        
        if region_seq is None:
            continue
        
        # Find START/STOP
        boundaries = find_start_stop(region_seq)
        
        if boundaries:
            start_pos, stop_pos = boundaries
            vgene_seq = region_seq[start_pos:stop_pos+3]  # Include stop codon
            
            # Translate
            try:
                protein_seq = vgene_seq.translate()
                
                # Basic filtering
                # Length check
                if len(protein_seq) < 90 or len(protein_seq) > 130:
                    continue
                
                # Check for premature stops (pseudogene indicator)
                # Allow only 1 stop at the end
                if str(protein_seq)[:-1].count('*') > 0:
                    continue
                
                # Remove terminal stop for output
                protein_seq_clean = str(protein_seq).rstrip('*')
                
                candidates.append({
                    'id': f"candidate_{idx}_{hit['qseqid']}",
                    'sequence': protein_seq_clean,
                    'contig': hit['sseqid'],
                    'coords': f"{start+start_pos}-{start+stop_pos}",
                    'strand': '+' if strand > 0 else '-',
                    'pident': hit['pident'],
                    'evalue': hit['evalue'],
                    'query': hit['qseqid']
                })
            
            except Exception as e:
                # Translation errors (e.g., incomplete codons)
                continue
    
    print(f"\n   ✅ Extracted {len(candidates)} valid candidates")
    
    # Save to FASTA
    print(f"\n💾 Saving to {output_fasta}")
    with open(output_fasta, 'w') as f:
        for cand in candidates:
            header = (f">{cand['id']} query={cand['query']} "
                     f"{cand['contig']}:{cand['coords']}({cand['strand']}) "
                     f"pident={cand['pident']:.1f}% evalue={cand['evalue']:.2e}")
            f.write(f"{header}\n")
            f.write(f"{cand['sequence']}\n")
    
    print(f"   ✅ Done")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"TBLASTN hits (raw): {len(hits_df)}")
    print(f"After quality filter: {len(filtered)}")
    print(f"Valid candidates extracted: {len(candidates)}")
    print(f"Extraction rate: {len(candidates)/len(filtered)*100:.1f}%")
    print(f"\nLength distribution:")
    lengths = [len(c['sequence']) for c in candidates]
    print(f"  Min: {min(lengths)} aa")
    print(f"  Max: {max(lengths)} aa")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f} aa")
    
    return candidates

if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).parent.parent
    DATA_DIR = PROJECT_DIR / "data"
    RESULTS_DIR = PROJECT_DIR / "results"
    
    # Species
    SPECIES = "myotis_lucifugus"
    SPECIES_RESULTS = RESULTS_DIR / SPECIES
    
    # Paths
    HITS_FILE = SPECIES_RESULTS / "tblastn_hits.txt"
    GENOME_DIR = DATA_DIR / "genomes" / SPECIES
    
    # Find genome
    genome_files = list(GENOME_DIR.rglob("*.fna"))
    if not genome_files:
        print(f"❌ No genome found")
        sys.exit(1)
    
    GENOME_FILE = genome_files[0]
    OUTPUT_FASTA = SPECIES_RESULTS / "candidates.fasta"
    
    # Check hits file exists
    if not HITS_FILE.exists():
        print(f"❌ TBLASTN hits file not found: {HITS_FILE}")
        print(f"   Run 06_run_tblastn.py first")
        sys.exit(1)
    
    print("="*70)
    print("EXTRACT V-GENE CANDIDATES")
    print("="*70)
    
    # Extract
    candidates = extract_candidates(HITS_FILE, GENOME_FILE, OUTPUT_FASTA)
    
    print(f"\n{'='*70}")
    print(f"✅ DONE")
    print(f"{'='*70}")
    print(f"Candidates saved to: {OUTPUT_FASTA}")
    print(f"\nNext step: Classify with CNN")
    print(f"  python scripts/predict_fasta.py {OUTPUT_FASTA} -o {SPECIES_RESULTS}/predictions.csv")