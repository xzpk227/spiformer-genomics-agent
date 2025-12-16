"""
Spliformer sidecar service.

POST /predict  — general mode, returns splicing scores as JSON
POST /motif    — motif mode, returns heatmap URLs (local static or S3 presigned)
GET  /health
"""
import os
import re
import glob
import logging
import tempfile
import shutil
import subprocess
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spliformer")

GENOME_DIR = os.environ.get("GENOME_DIR", "/ref")
ANNO_DIR = "/app/Spliformer/reference"
REPORTS_BUCKET = os.environ.get("REPORTS_BUCKET")
PRESIGNED_URL_EXPIRY = 60 * 60 * 24  # 24h
LOCAL_REPORTS = "/app/reports/spliformer"

app = FastAPI(title="Spliformer Service")

# Serve heatmaps locally when not using S3
if not REPORTS_BUCKET:
    os.makedirs(LOCAL_REPORTS, exist_ok=True)
    app.mount("/heatmaps", StaticFiles(directory=LOCAL_REPORTS), name="heatmaps")


# --- helpers ---

def genome_path(assembly: str) -> str:
    name = "hg19.fa" if assembly == "hg19" else "hg38.fa"
    return os.path.join(GENOME_DIR, name)


def anno_path(assembly: str) -> str:
    name = "hg19anno.txt" if assembly == "hg19" else "hg38anno.txt"
    return os.path.join(ANNO_DIR, name)


def write_vcf(path: str, chrom_vcf: str, pos: int, ref: str, alt: str):
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.1\n")
        for i in list(range(1, 23)) + ["X", "Y", "MT"]:
            f.write(f"##contig=<ID={i}>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write(f"{chrom_vcf}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")


def upload_to_s3(local_path: str, key: str) -> str:
    s3 = boto3.client("s3")
    s3.upload_file(local_path, REPORTS_BUCKET, key,
                   ExtraArgs={"ContentType": "image/png"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": REPORTS_BUCKET, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )
    return url


def check_genome(assembly: str):
    ref_fa = genome_path(assembly)
    if not os.path.exists(ref_fa):
        raise HTTPException(
            status_code=503,
            detail=f"Reference genome not found at {ref_fa}. Mount a {assembly} FASTA to {GENOME_DIR}.",
        )
    return ref_fa


# --- models ---

class PredictRequest(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genome: str = "hg38"


class MotifRequest(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genome: str = "hg38"
    n_motifs: int = 10


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    ref_fa = check_genome(req.genome)
    anno = anno_path(req.genome)
    chrom = req.chrom if req.chrom.startswith("chr") else f"chr{req.chrom}"
    chrom_vcf = chrom[3:]

    with tempfile.TemporaryDirectory() as tmpdir:
        vcf_in = os.path.join(tmpdir, "input.vcf")
        vcf_out = os.path.join(tmpdir, "output.vcf")
        write_vcf(vcf_in, chrom_vcf, req.pos, req.ref, req.alt)

        cmd = ["spliformer", "-T", "general", "-I", vcf_in, "-O", vcf_out, "-R", ref_fa, "-A", anno]
        logger.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Spliformer timed out")

        if result.returncode != 0:
            logger.error("stdout: %s", result.stdout)
            logger.error("stderr: %s", result.stderr)
            raise HTTPException(status_code=500, detail=f"Spliformer error: {result.stderr[:500]}")

        scores = parse_output_vcf(vcf_out)
        return {"variant": f"{chrom}:{req.pos}:{req.ref}:{req.alt}", "genome": req.genome, "scores": scores}


@app.post("/motif")
def motif(req: MotifRequest):
    """Run Spliformer motif mode and return heatmap image URLs."""
    ref_fa = check_genome(req.genome)
    anno = anno_path(req.genome)
    chrom = req.chrom if req.chrom.startswith("chr") else f"chr{req.chrom}"
    chrom_vcf = chrom[3:]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_key = f"{chrom}_{req.pos}_{req.ref}_{req.alt}"

    with tempfile.TemporaryDirectory() as tmpdir:
        vcf_in = os.path.join(tmpdir, "input.vcf")
        write_vcf(vcf_in, chrom_vcf, req.pos, req.ref, req.alt)

        cmd = [
            "spliformer", "-T", "motif",
            "-I", vcf_in,
            "-R", ref_fa,
            "-A", anno,
            "-N", str(req.n_motifs),
        ]
        logger.info("Running motif: %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=tmpdir)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Spliformer motif mode timed out")

        if result.returncode != 0:
            logger.error("stdout: %s", result.stdout)
            logger.error("stderr: %s", result.stderr)
            raise HTTPException(status_code=500, detail=f"Spliformer motif error: {result.stderr[:500]}")

        logger.info("Motif stdout: %s", result.stdout)
        logger.info("Motif stderr: %s", result.stderr)

        # Find generated PNG heatmaps
        pngs = glob.glob(os.path.join(tmpdir, "motif_result", "**", "*.png"), recursive=True)
        if not pngs:
            pngs = glob.glob(os.path.join(tmpdir, "**", "*.png"), recursive=True)
        logger.info("PNG search in %s, found: %s", tmpdir, pngs)

        if not pngs:
            return {"variant": f"{chrom}:{req.pos}:{req.ref}:{req.alt}", "images": [],
                    "message": "Motif mode ran but no heatmap images were generated."}

        image_urls = []
        for png in pngs:
            fname = f"{variant_key}_{timestamp}_{os.path.basename(png)}"
            if REPORTS_BUCKET:
                try:
                    key = f"spliformer/motifs/{fname}"
                    url = upload_to_s3(png, key)
                    image_urls.append({"filename": fname, "url": url, "storage": "s3"})
                except (BotoCoreError, ClientError) as e:
                    logger.error("S3 upload failed: %s", e)
                    image_urls.append({"filename": fname, "url": None, "error": str(e)})
            else:
                dest = os.path.join(LOCAL_REPORTS, fname)
                shutil.copy(png, dest)
                image_urls.append({
                    "filename": fname,
                    "url": f"/heatmaps/{fname}",
                    "storage": "local",
                })

        return {
            "variant": f"{chrom}:{req.pos}:{req.ref}:{req.alt}",
            "genome": req.genome,
            "images": image_urls,
        }


# --- VCF parsing ---

def parse_output_vcf(vcf_path: str) -> dict:
    if not os.path.exists(vcf_path):
        return {}
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            info = parts[7]
            m = re.search(r"Spliformer=([^\s;]+)", info)
            raw = m.group(1) if m else info
            fields = raw.split("|")
            if len(fields) < 6:
                return {"raw": raw}
            return {
                "alleles": fields[0],
                "gene": fields[1],
                "acceptor_gain": _parse_score(fields[2]),
                "acceptor_loss": _parse_score(fields[3]),
                "donor_gain": _parse_score(fields[4]),
                "donor_loss": _parse_score(fields[5]),
                "raw": raw,
            }
    return {}


def _parse_score(field: str) -> dict:
    try:
        dist, score = field.split(":")
        return {"distance": int(dist), "score": float(score)}
    except Exception:
        return {"raw": field}
