#!/usr/bin/env python3
"""
Verify V-Gene Classifier v2.0.0 dependencies.
"""

import sys

def check_package(package_name, min_version=None):
    """Check if package is installed with minimum version."""
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"❌ {package_name} {version} (requires >={min_version})")
                return False

        print(f"✅ {package_name} {version}")
        return True
    except ImportError:
        print(f"❌ {package_name} NOT INSTALLED")
        return False

def check_system_tool(command, name):
    """Check if system tool is available."""
    import shutil
    if shutil.which(command):
        print(f"✅ {name} installed")
        return True
    else:
        print(f"❌ {name} NOT FOUND (install separately)")
        return False

def main():
    print("=" * 70)
    print("V-GENE CLASSIFIER v2.0.0 - DEPENDENCY CHECK")
    print("=" * 70)
    print()

    print("📦 Python Packages:")
    print("-" * 70)
    all_ok = True
    all_ok &= check_package('torch', '2.0.0')
    all_ok &= check_package('Bio', '1.81')  # biopython
    all_ok &= check_package('pandas', '2.0.0')
    all_ok &= check_package('numpy', '1.24.0')
    all_ok &= check_package('sklearn', '1.3.0')  # scikit-learn
    all_ok &= check_package('matplotlib', '3.7.0')
    all_ok &= check_package('tqdm', '4.65.0')

    print()
    print("🔧 System Tools:")
    print("-" * 70)
    all_ok &= check_system_tool('makeblastdb', 'BLAST+')
    all_ok &= check_system_tool('tblastn', 'BLAST+')

    print()
    print("🔬 Optional Tools:")
    print("-" * 70)
    check_system_tool('clustalo', 'ClustalO (for phylogenetic validation)')

    print()
    print("💻 GPU Check:")
    print("-" * 70)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   PyTorch CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available (CPU only)")
    except:
        pass

    print()
    print("=" * 70)
    if all_ok:
        print("✅ ALL REQUIRED DEPENDENCIES INSTALLED")
    else:
        print("❌ SOME DEPENDENCIES MISSING - See errors above")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    main()
