import json
import requests
from langchain.tools import tool


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
