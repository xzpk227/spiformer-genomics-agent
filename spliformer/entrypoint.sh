#!/bin/bash
set -e

HG38_FA="/ref/hg38.fa"
HG38_GZ="/ref/hg38.fa.gz"
HG19_FA="/ref/hg19.fa"
HG19_GZ="/ref/hg19.fa.gz"

download_genome() {
    local fa="$1"
    local gz="$2"
    local url="$3"
    local name="$4"

    if [ -f "$fa" ]; then
        echo "[spliformer] $name already exists, skipping download."
        return
    fi

    echo "[spliformer] $name not found. Downloading (~3GB, this will take a while)..."
    curl -L --retry 3 --retry-delay 5 --progress-bar -o "$gz" "$url"
    echo "[spliformer] Decompressing $name..."
    gunzip "$gz"
    echo "[spliformer] $name ready at $fa"
}

# Download whichever genomes are missing (skipped in cloud if EFS is pre-loaded)
download_genome \
    "$HG38_FA" "$HG38_GZ" \
    "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz" \
    "hg38"

# Uncomment to also bootstrap hg19:
download_genome \
    "$HG19_FA" "$HG19_GZ" \
    "https://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz" \
    "hg19"

echo "[spliformer] Starting API server..."
exec uvicorn app:app --host 0.0.0.0 --port 5001
