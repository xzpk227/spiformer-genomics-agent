import json
import os
import requests
from langchain.tools import tool

SPLIFORMER_URL = os.environ.get("SPLIFORMER_URL", "http://spliformer:5001")


@tool
def predict_splicing(gene_symbol: str, variant: str = "") -> str:
    """
    Predict splicing impact for a gene or variant using SpliceAI-lookup API.
    variant format: 'chr:pos:ref:alt' e.g. '7:117548628:A:G'
    """
    try:
        if not variant:
            return (
                f"Splicing prediction for {gene_symbol}: Please provide a variant in format "
                "'chr:pos:ref:alt' for SpliceAI scoring. "
                "Without a specific variant, splicing analysis requires variant data from GWAS or sequencing."
            )
        # SpliceAI-lookup public API
        url = "https://spliceailookup-api.broadinstitute.org/spliceai/"
        params = {"hg": "38", "variant": variant, "distance": 500, "mask": 0}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        scores = data.get("scores", [])
        if not scores:
            return f"No SpliceAI scores returned for variant {variant}"
        result = [f"SpliceAI scores for {variant}:"]
        for s in scores:
            result.append(
                f"  Gene: {s.get('gene_name')} | "
                f"DS_AG: {s.get('DS_AG'):.3f} | DS_AL: {s.get('DS_AL'):.3f} | "
                f"DS_DG: {s.get('DS_DG'):.3f} | DS_DL: {s.get('DS_DL'):.3f} | "
                f"Max delta: {max(s.get('DS_AG',0), s.get('DS_AL',0), s.get('DS_DG',0), s.get('DS_DL',0)):.3f}"
            )
        return "\n".join(result)
    except Exception as e:
        return f"Splicing prediction error: {str(e)}"


@tool
def run_enrichment_analysis(gene_list: str) -> str:
    """
    Run GO and KEGG pathway enrichment analysis on a comma-separated list of gene symbols.
    Example input: 'PTPRN2,SOD1,FUS,TDP43,C9orf72'
    """
    try:
        import gseapy as gp
        genes = [g.strip() for g in gene_list.split(",") if g.strip()]
        if len(genes) < 3:
            return "Please provide at least 3 gene symbols for enrichment analysis."

        results_summary = []

        # GO Biological Process
        enr_go = gp.enrichr(
            gene_list=genes,
            gene_sets=["GO_Biological_Process_2023"],
            organism="human",
            outdir=None,
            verbose=False,
        )
        top_go = enr_go.results.head(5)
        results_summary.append("Top GO Biological Process terms:")
        for _, row in top_go.iterrows():
            results_summary.append(
                f"  {row['Term']} | Adj p-val: {row['Adjusted P-value']:.4f} | Overlap: {row['Overlap']}"
            )

        # KEGG Pathways
        enr_kegg = gp.enrichr(
            gene_list=genes,
            gene_sets=["KEGG_2021_Human"],
            organism="human",
            outdir=None,
            verbose=False,
        )
        top_kegg = enr_kegg.results.head(5)
        results_summary.append("\nTop KEGG Pathways:")
        for _, row in top_kegg.iterrows():
            results_summary.append(
                f"  {row['Term']} | Adj p-val: {row['Adjusted P-value']:.4f} | Overlap: {row['Overlap']}"
            )

        return "\n".join(results_summary)
    except Exception as e:
        return f"Enrichment analysis error: {str(e)}"


@tool
def predict_splicing_spliformer(variant: str, genome: str = "hg38") -> str:
    """
    Predict splicing impact using the local Spliformer deep-learning model.
    Requires a reference genome mounted in the spliformer service.
    variant format: 'chr:pos:ref:alt' e.g. 'chr2:179642185:G:A'
    genome: 'hg38' (default) or 'hg19'
    """
    try:
        # Check service is available
        try:
            requests.get(f"{SPLIFORMER_URL}/health", timeout=3).raise_for_status()
        except Exception:
            return (
                "Spliformer service is not available. "
                "Ensure the spliformer container is running and a reference genome is mounted. "
                "See README for setup instructions."
            )

        parts = variant.replace(" ", "").split(":")
        if len(parts) != 4:
            return "Invalid variant format. Use 'chr:pos:ref:alt' e.g. 'chr2:179642185:G:A'"

        chrom, pos, ref, alt = parts
        payload = {"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt, "genome": genome}
        r = requests.post(f"{SPLIFORMER_URL}/predict", json=payload, timeout=130)

        if r.status_code == 503:
            return r.json().get("detail", "Reference genome not mounted.")
        r.raise_for_status()

        data = r.json()
        scores = data.get("scores", {})
        if not scores:
            return f"No Spliformer scores returned for {variant}"

        gene = scores.get("gene", "unknown")
        ag = scores.get("acceptor_gain", {})
        al = scores.get("acceptor_loss", {})
        dg = scores.get("donor_gain", {})
        dl = scores.get("donor_loss", {})

        def fmt(s: dict) -> str:
            if "score" in s:
                return f"{s['score']:.3f} (dist {s['distance']})"
            return s.get("raw", "N/A")

        all_scores = [
            s.get("score", 0) for s in [ag, al, dg, dl] if isinstance(s, dict) and "score" in s
        ]
        max_score = max(all_scores) if all_scores else 0
        impact = "HIGH" if max_score >= 0.5 else "MODERATE" if max_score >= 0.2 else "LOW"

        return (
            f"Spliformer scores for {variant} ({genome}):\n"
            f"  Gene: {gene}\n"
            f"  Acceptor gain: {fmt(ag)}\n"
            f"  Acceptor loss: {fmt(al)}\n"
            f"  Donor gain:    {fmt(dg)}\n"
            f"  Donor loss:    {fmt(dl)}\n"
            f"  Max score: {max_score:.3f} → Impact: {impact}"
        )
    except Exception as e:
        return f"Spliformer prediction error: {str(e)}"


@tool
def spliformer_motif(variant: str, genome: str = "hg38", n_motifs: int = 10) -> str:
    """
    Generate Spliformer attention weight score (AWS) heatmaps for splicing motifs.
    Shows which sequence motifs the model attends to in wildtype vs variant sequences.
    variant format: 'chr:pos:ref:alt' e.g. 'chr2:179642185:G:A'
    genome: 'hg38' (default) or 'hg19'
    n_motifs: number of motifs to visualize (default 10, max 40)
    """
    try:
        try:
            requests.get(f"{SPLIFORMER_URL}/health", timeout=3).raise_for_status()
        except Exception:
            return "Spliformer service is not available."

        parts = variant.replace(" ", "").split(":")
        if len(parts) != 4:
            return "Invalid variant format. Use 'chr:pos:ref:alt'"

        chrom, pos, ref, alt = parts
        payload = {"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                   "genome": genome, "n_motifs": n_motifs}
        r = requests.post(f"{SPLIFORMER_URL}/motif", json=payload, timeout=200)

        if r.status_code == 503:
            return r.json().get("detail", "Reference genome not mounted.")
        r.raise_for_status()

        data = r.json()
        images = data.get("images", [])
        if not images:
            return data.get("message", "No heatmaps generated.")

        lines = [f"Spliformer motif heatmaps for {variant} ({genome}):"]
        for img in images:
            url = img.get("url")
            storage = img.get("storage", "")
            if url:
                prefix = "http://localhost:5001" if storage == "local" else ""
                lines.append(f"  {img['filename']}: {prefix}{url}")
            else:
                lines.append(f"  {img['filename']}: upload failed — {img.get('error')}")
        return "\n".join(lines)
    except Exception as e:
        return f"Spliformer motif error: {str(e)}"
