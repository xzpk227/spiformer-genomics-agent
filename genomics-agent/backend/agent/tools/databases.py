import os
import requests
from langchain.tools import tool


ENSEMBL_BASE = "https://rest.ensembl.org"



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
