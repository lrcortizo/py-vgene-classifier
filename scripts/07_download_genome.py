#!/usr/bin/env python3
"""
Download genome assembly for a target species.

This script downloads a genome assembly from NCBI using the datasets CLI tool
or direct FTP download, and prepares it for TBLASTN search.

Usage:
    python scripts/07_download_genome.py \
        --accession GCF_000147115.1 \
        --output-dir data/genomes/myotis_lucifugus
"""

import argparse
import subprocess
import gzip
import shutil
import urllib.request
import json
from pathlib import Path
from Bio import SeqIO


def download_with_datasets(accession: str, output_dir: Path) -> bool:
    """
    Try to download using NCBI datasets CLI tool.

    Returns True if successful, False otherwise.
    """
    print(f"\n🔧 Attempting download with NCBI datasets...")

    # Check if datasets is installed
    try:
        result = subprocess.run(
            ["datasets", "--version"],
            capture_output=True,
            text=True
        )
        print(f"   ✅ datasets CLI found: {result.stdout.strip()}")
    except FileNotFoundError:
        print(f"   ❌ datasets CLI not found")
        return False

    # Download
    try:
        print(f"   📥 Downloading {accession}...")

        cmd = [
            "datasets", "download", "genome", "accession",
            accession,
            "--include", "genome",
            "--filename", str(output_dir / "genome.zip")
        ]

        subprocess.run(cmd, check=True)

        # Unzip
        print(f"   📦 Extracting...")
        subprocess.run(
            ["unzip", "-o", str(output_dir / "genome.zip"), "-d", str(output_dir)],
            check=True,
            capture_output=True
        )

        # Find FASTA file
        fasta_files = list(output_dir.glob("**/*.fna"))
        if fasta_files:
            final_fasta = output_dir / f"{accession}.fna"
            shutil.copy(fasta_files[0], final_fasta)
            print(f"   ✅ Genome saved: {final_fasta}")
            return True

        return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_with_wget(accession: str, output_dir: Path) -> bool:
    """
    Download directly from NCBI FTP using wget.

    Returns True if successful, False otherwise.
    """
    print(f"\n🌐 Attempting download via FTP...")

    # Construct FTP URL
    # Example: GCF_000147115.1 -> GCF/000/147/115/GCF_000147115.1
    parts = accession.split('_')
    prefix = parts[0]  # GCF or GCA
    number = parts[1].split('.')[0]  # 000147115

    # Split number into groups
    group1 = number[0:3]
    group2 = number[3:6]
    group3 = number[6:9]

    base_url = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{prefix}/{group1}/{group2}/{group3}"

    # Find exact directory (need to match suffix)
    print(f"   🔍 Looking for assembly at: {base_url}/")

    # Get assembly metadata
    try:
        api_url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{accession}"
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read())
            assembly_name = data['reports'][0]['assembly_info']['assembly_name']
            suffixes = [f"_{assembly_name}", "", "_latest"]
    except:
        suffixes = ["_Myoluc2.0", "", "_latest"]  # Fallback with common patterns

    for suffix in suffixes:
        url = f"{base_url}/{accession}{suffix}/{accession}{suffix}_genomic.fna.gz"
        output_gz = output_dir / f"{accession}_genomic.fna.gz"

        try:
            print(f"   📥 Trying: {url}")

            cmd = ["wget", "-q", "-O", str(output_gz), url]
            result = subprocess.run(cmd)

            if result.returncode == 0 and output_gz.exists():
                # Decompress
                print(f"   📦 Decompressing...")
                output_fna = output_dir / f"{accession}.fna"

                with gzip.open(output_gz, 'rt') as f_in:
                    with open(output_fna, 'w') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                output_gz.unlink()  # Remove .gz

                print(f"   ✅ Genome saved: {output_fna}")
                return True

        except Exception as e:
            print(f"   ⚠️  Failed: {e}")
            continue

    return False


def get_genome_stats(fasta_file: Path):
    """Print genome statistics."""
    print(f"\n📊 Genome Statistics:")

    records = list(SeqIO.parse(fasta_file, "fasta"))

    total_length = sum(len(rec.seq) for rec in records)
    lengths = [len(rec.seq) for rec in records]

    print(f"   Contigs: {len(records)}")
    print(f"   Total length: {total_length:,} bp ({total_length/1e6:.1f} Mb)")
    print(f"   Longest contig: {max(lengths):,} bp")
    print(f"   N50: {calculate_n50(lengths):,} bp")
    print(f"   GC content: {calculate_gc(records):.1f}%")


def calculate_n50(lengths):
    """Calculate N50."""
    sorted_lengths = sorted(lengths, reverse=True)
    total = sum(sorted_lengths)
    cumsum = 0

    for length in sorted_lengths:
        cumsum += length
        if cumsum >= total / 2:
            return length
    return 0


def calculate_gc(records):
    """Calculate GC content."""
    total_gc = 0
    total_length = 0

    for rec in records:
        seq = str(rec.seq).upper()
        total_gc += seq.count('G') + seq.count('C')
        total_length += len(seq)

    return (total_gc / total_length * 100) if total_length > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Download genome assembly from NCBI"
    )
    parser.add_argument(
        "--accession",
        type=str,
        required=True,
        help="NCBI assembly accession (e.g., GCF_000147115.1)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for genome files"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "datasets", "wget"],
        default="auto",
        help="Download method (default: auto - try both)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DOWNLOAD GENOME ASSEMBLY")
    print("=" * 70)
    print(f"\n📋 Accession: {args.accession}")
    print(f"📁 Output: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Try download methods
    success = False

    if args.method in ["auto", "datasets"]:
        success = download_with_datasets(args.accession, args.output_dir)

    if not success and args.method in ["auto", "wget"]:
        success = download_with_wget(args.accession, args.output_dir)

    if not success:
        print("\n❌ Failed to download genome")
        print("\n💡 Manual download:")
        print(f"   1. Go to: https://www.ncbi.nlm.nih.gov/assembly/{args.accession}")
        print(f"   2. Click 'Download Assemblies'")
        print(f"   3. Download genome FASTA")
        print(f"   4. Save to: {args.output_dir}/{args.accession}.fna")
        return

    # Find downloaded file
    fasta_file = args.output_dir / f"{args.accession}.fna"

    if not fasta_file.exists():
        fasta_files = list(args.output_dir.glob("*.fna"))
        if fasta_files:
            fasta_file = fasta_files[0]

    if fasta_file.exists():
        get_genome_stats(fasta_file)

        print("\n" + "=" * 70)
        print("✅ DONE")
        print("=" * 70)
        print(f"\nGenome ready: {fasta_file}")
        print("\nNext step:")
        print(f"  python scripts/08_run_tblastn.py \\")
        print(f"      --genome {fasta_file} \\")
        print(f"      --query data/raw/positive/human_vgenes.fasta \\")
        print(f"      --output-dir results/tblastn")
    else:
        print("\n❌ Genome file not found after download")


if __name__ == "__main__":
    main()
