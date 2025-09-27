import os
import requests
from langchain.tools import tool


ENSEMBL_BASE = "https://rest.ensembl.org"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GWAS_BASE = "https://www.ebi.ac.uk/gwas/rest/api"


@tool
def query_ensembl(gene_symbol: str) -> str:
    """Query Ensembl REST API for gene structure, transcripts, and annotations."""
    try:
        headers = {"Content-Type": "application/json"}
        # Lookup gene by symbol
        url = f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}"
        r = requests.get(url, headers=headers, params={"expand": 1}, timeout=15)
        r.raise_for_status()
        data = r.json()

        transcripts = data.get("Transcript", [])
        summary = (
            f"Gene: {data.get('display_name')} ({data.get('id')})\n"
            f"Description: {data.get('description', 'N/A')}\n"
            f"Location: Chr{data.get('seq_region_name')}:{data.get('start')}-{data.get('end')} ({data.get('strand')})\n"
            f"Biotype: {data.get('biotype')}\n"
            f"Transcripts: {len(transcripts)}\n"
        )
        for t in transcripts[:5]:
            summary += f"  - {t.get('id')}: {t.get('biotype')}, {t.get('length')} bp\n"
        return summary
    except Exception as e:
        return f"Ensembl query error: {str(e)}"


@tool
def query_ncbi_gene(gene_symbol: str) -> str:
    """Query NCBI Gene database for gene summaries and functional information."""
    try:
        api_key = os.environ.get("NCBI_API_KEY", "")
        # Search for gene ID
        search_url = f"{NCBI_BASE}/esearch.fcgi"
        params = {
            "db": "gene", "term": f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
            "retmode": "json", "retmax": 1, "api_key": api_key
        }
        r = requests.get(search_url, params=params, timeout=15)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No NCBI Gene entry found for {gene_symbol}"

        # Fetch summary
        summary_url = f"{NCBI_BASE}/esummary.fcgi"
        r2 = requests.get(summary_url, params={"db": "gene", "id": ids[0], "retmode": "json", "api_key": api_key}, timeout=15)
        r2.raise_for_status()
        result = r2.json().get("result", {}).get(ids[0], {})

        return (
            f"NCBI Gene ID: {ids[0]}\n"
            f"Name: {result.get('name')}\n"
            f"Full Name: {result.get('description')}\n"
            f"Summary: {result.get('summary', 'N/A')[:800]}\n"
            f"Chromosome: {result.get('chromosome')}\n"
            f"Location: {result.get('maplocation')}\n"
        )
    except Exception as e:
        return f"NCBI Gene query error: {str(e)}"


@tool
def query_gwas_catalog(gene_or_disease: str) -> str:
    """Query GWAS Catalog for disease-gene associations and significant SNPs."""
    try:
        url = f"{GWAS_BASE}/associations/search"
        params = {"q": gene_or_disease, "size": 10}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        associations = data.get("_embedded", {}).get("associations", [])

        if not associations:
            return f"No GWAS associations found for '{gene_or_disease}'"

        results = [f"GWAS Catalog associations for '{gene_or_disease}':"]
        for assoc in associations[:8]:
            snp = assoc.get("strongestAllele", [{}])
            trait = assoc.get("efoTraits", [{}])
            pval = assoc.get("pvalue")
            snp_id = snp[0].get("strongestRiskAllele", "N/A") if snp else "N/A"
            trait_name = trait[0].get("trait", "N/A") if trait else "N/A"
            results.append(f"  SNP: {snp_id} | Trait: {trait_name} | p-value: {pval}")
        return "\n".join(results)
    except Exception as e:
        return f"GWAS Catalog query error: {str(e)}"
