"""
Download genome for a target species
"""
import subprocess
import sys
from pathlib import Path
import zipfile

def download_genome(accession, species_name, output_dir):
    """
    Download genome from NCBI using datasets CLI
    
    Args:
        accession: NCBI accession (e.g., 'GCF_000147115.1')
        species_name: Species name for folder (e.g., 'myotis_lucifugus')
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    species_dir = output_dir / species_name
    species_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"DOWNLOADING GENOME")
    print(f"{'='*70}")
    print(f"Species: {species_name}")
    print(f"Accession: {accession}")
    print(f"Output: {species_dir}")
    
    zip_file = species_dir / f"{species_name}.zip"
    
    # Download using NCBI datasets
    cmd = [
        "datasets", "download", "genome", "accession", accession,
        "--filename", str(zip_file)
    ]
    
    print(f"\n📥 Downloading... (this may take several minutes)")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading: {e}")
        return None
    
    # Unzip
    print(f"\n📦 Extracting...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(species_dir)
    
    print(f"\n✅ Genome downloaded to {species_dir}")
    
    # Find the FASTA file
    fasta_files = list(species_dir.rglob("*.fna"))
    if fasta_files:
        main_fasta = fasta_files[0]
        print(f"📄 Main genome file: {main_fasta}")
        print(f"   Size: {main_fasta.stat().st_size / 1024 / 1024:.1f} MB")
        return main_fasta
    else:
        print("⚠️  No FASTA file found in download")
        return None

if __name__ == "__main__":
    # Test with little brown bat (Myotis lucifugus)
    DATA_DIR = Path(__file__).parent.parent / "data" / "genomes"
    
    print("="*70)
    print("GENOME DOWNLOADER")
    print("="*70)
    
    # Example species (puedes cambiar)
    genome = download_genome(
        accession="GCF_000147115.1",
        species_name="myotis_lucifugus",
        output_dir=DATA_DIR
    )
    
    if genome:
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS")
        print(f"{'='*70}")
        print(f"Genome ready at: {genome}")
        print(f"\nNext step: python scripts/06_run_tblastn.py")
    else:
        print(f"\n{'='*70}")
        print(f"❌ FAILED")
        print(f"{'='*70}")