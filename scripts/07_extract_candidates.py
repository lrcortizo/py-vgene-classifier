"""
Extract candidate V-gene sequences from BLAST hits
"""
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from pathlib import Path
import sys
import re

def load_genome(genome_file):
    """Load genome into memory as dictionary"""
    print(f"\n📖 Loading genome: {genome_file}")
    print(f"   (this may take a minute for large genomes...)")
    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
    print(f"   ✅ Loaded {len(genome)} contigs/chromosomes")
    return genome

def extract_region(genome, contig_id, start, end, strand):
    """Extract a genomic region"""
    if contig_id not in genome:
        return None
    
    # Extract region (convert to 0-indexed)
    seq = genome[contig_id].seq[start-1:end]
    
    # Reverse complement if negative strand
    if strand < 0:
        seq = seq.reverse_complement()
    
    return seq

def find_start_stop(dna_seq, min_length=200, max_length=500):
    """Find START (ATG) and STOP codons in sequence"""
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
    best_pair = None
    best_score = float('inf')
    
    for start in start_codons:
        for stop in stop_codons:
            if stop > start:
                length = stop - start
                # Check if in-frame
                if (stop - start) % 3 != 0:
                    continue
                # Check length
                if min_length <= length <= max_length:
                    # Prefer length close to 330 (typical V-gene)
                    score = abs(length - 330)
                    if score < best_score:
                        best_pair = (start, stop)
                        best_score = score
    
    return best_pair

def remove_leader(protein_seq):
    """
    Remove leader sequence from V-gene
    Returns: (cleaned_sequence, found_pattern) or (None, None) if no pattern found
    """
    protein_str = str(protein_seq)
    
    # Framework 1 patterns (start of mature V-domain)
    fr1_patterns = [
        'QVQLVQSG', 'QVQLVESG', 'EVQLVESG', 'QVQLQQSG',
        'EVQLLESG', 'QVQLQESG', 'QVTLKESG', 'QVQLKESG',
        'QITLKESG', 'DVQLVESG', 'QVQLLESGG', 'EVQLLESGG',
        'QSVEESGG', 'QVQLQESGG', 'QVQLVESGG'
    ]
    
    # Try to find FR1 pattern
    for pattern in fr1_patterns:
        pos = protein_str.find(pattern)
        if pos != -1 and pos < 40:  # Leader should be <40 aa
            return protein_str[pos:], pattern
    
    # If no pattern found, check if sequence already starts with Q/E (typical FR1)
    if protein_str[0] in ['Q', 'E', 'D']:
        # Might already be trimmed
        return protein_str, "already_trimmed"
    
    return None, None

def extract_candidates(hits_file, genome_file, output_fasta, extend=500):
    """Extract candidate V-gene sequences from TBLASTN hits"""
    
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
    
    # Filter by quality
    filtered = hits_df[
        (hits_df['pident'] >= 60) &
        (hits_df['evalue'] <= 1e-5)
    ]
    print(f"   After filtering (pident>=60%, evalue<=1e-5): {len(filtered)}")
    
    print(f"\n🔬 Extracting candidate sequences...")
    
    candidates = []
    stats = {
        'total_processed': 0,
        'found_start_stop': 0,
        'translation_ok': 0,
        'found_fr1_pattern': 0,
        'length_ok': 0,
        'no_premature_stops': 0,
        'final_candidates': 0
    }
    
    for idx, hit in filtered.iterrows():
        stats['total_processed'] += 1
        
        # Progress
        if idx % 100 == 0:
            print(f"   Progress: {idx}/{len(filtered)}", end='\r')
        
        # Determine strand
        strand = 1 if hit['sstart'] < hit['send'] else -1
        start = min(hit['sstart'], hit['send']) - extend
        end = max(hit['sstart'], hit['send']) + extend
        start = max(1, start)
        
        # Extract region
        region_seq = extract_region(genome, hit['sseqid'], start, end, strand)
        if region_seq is None:
            continue
        
        # Find START/STOP
        boundaries = find_start_stop(region_seq)
        if not boundaries:
            continue
        
        stats['found_start_stop'] += 1
        
        start_pos, stop_pos = boundaries
        vgene_seq = region_seq[start_pos:stop_pos+3]
        
        # Translate
        try:
            protein_seq = vgene_seq.translate()
            stats['translation_ok'] += 1
        except:
            continue
        
        # Remove terminal stop
        protein_str = str(protein_seq).rstrip('*')
        
        # Try to remove leader
        cleaned_seq, pattern = remove_leader(protein_str)
        
        if cleaned_seq is None:
            # No FR1 pattern found, keep original
            cleaned_seq = protein_str
            pattern = "none"
        else:
            stats['found_fr1_pattern'] += 1
        
        # Length check (after leader removal)
        if len(cleaned_seq) < 85 or len(cleaned_seq) > 135:
            continue
        
        stats['length_ok'] += 1
        
        # Check for premature stops
        if '*' in cleaned_seq:
            continue
        
        stats['no_premature_stops'] += 1
        stats['final_candidates'] += 1
        
        candidates.append({
            'id': f"candidate_{idx}_{hit['qseqid']}",
            'sequence': cleaned_seq,
            'contig': hit['sseqid'],
            'coords': f"{start+start_pos}-{start+stop_pos}",
            'strand': '+' if strand > 0 else '-',
            'pident': hit['pident'],
            'evalue': hit['evalue'],
            'query': hit['qseqid'],
            'fr1_pattern': pattern,
            'length': len(cleaned_seq)
        })
    
    print(f"\n   ✅ Extracted {len(candidates)} valid candidates")
    
    # Save to FASTA
    if candidates:
        print(f"\n💾 Saving to {output_fasta}")
        with open(output_fasta, 'w') as f:
            for cand in candidates:
                header = (f">{cand['id']} query={cand['query']} "
                         f"{cand['contig']}:{cand['coords']}({cand['strand']}) "
                         f"pident={cand['pident']:.1f}% evalue={cand['evalue']:.2e} "
                         f"fr1={cand['fr1_pattern']} len={cand['length']}")
                f.write(f"{header}\n")
                f.write(f"{cand['sequence']}\n")
        print(f"   ✅ Done")
    else:
        print(f"\n⚠️  No valid candidates found")
        print(f"\n📊 Extraction pipeline stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        return []
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"EXTRACTION PIPELINE STATS")
    print(f"{'='*70}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"TBLASTN hits (raw): {len(hits_df)}")
    print(f"After quality filter: {len(filtered)}")
    print(f"Valid candidates extracted: {len(candidates)}")
    if len(filtered) > 0:
        print(f"Extraction rate: {len(candidates)/len(filtered)*100:.1f}%")
    
    if candidates:
        lengths = [c['length'] for c in candidates]
        print(f"\nLength distribution:")
        print(f"  Min: {min(lengths)} aa")
        print(f"  Max: {max(lengths)} aa")
        print(f"  Mean: {sum(lengths)/len(lengths):.1f} aa")
        
        # FR1 pattern distribution
        patterns = {}
        for c in candidates:
            p = c['fr1_pattern']
            patterns[p] = patterns.get(p, 0) + 1
        
        print(f"\nFR1 patterns found:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")
    
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
    
    if candidates:
        print(f"\n{'='*70}")
        print(f"✅ DONE")
        print(f"{'='*70}")
        print(f"Candidates saved to: {OUTPUT_FASTA}")
        print(f"\nNext step: Classify with CNN")
        print(f"  python scripts/predict_fasta.py {OUTPUT_FASTA} -o {SPECIES_RESULTS}/predictions.csv")
    else:
        print(f"\n{'='*70}")
        print(f"⚠️  NO CANDIDATES EXTRACTED")
        print(f"{'='*70}")
        print(f"Check the extraction pipeline stats above to see where candidates were filtered out.")