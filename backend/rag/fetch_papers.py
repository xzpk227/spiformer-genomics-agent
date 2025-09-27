"""
Fetch full-text papers from PubMed Central (PMC) and save as text files for ingestion.

Usage:
    python rag/fetch_papers.py --query "PTPRN2 ALS" --max 50 --outdir /app/papers

Requirements: NCBI_API_KEY in .env (optional but increases rate limit from 3 to 10 req/s)
"""
import os
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
API_KEY = os.environ.get("NCBI_API_KEY", "")


def search_pmc(query: str, max_results: int) -> list[str]:
    """Search PMC and return a list of PMC IDs."""
    print(f"Searching PMC for: '{query}' (max {max_results} results)...")
    params = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "y",
    }
    if API_KEY:
        params["api_key"] = API_KEY

    r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    print(f"Found {len(ids)} articles.")
    return ids


def fetch_full_text(pmc_id: str) -> str | None:
    """Fetch full text XML from PMC OAI and extract plain text."""
    try:
        params = {
            "verb": "GetRecord",
            "identifier": f"oai:pubmedcentral.nih.gov:{pmc_id}",
            "metadataPrefix": "pmc",
        }
        r = requests.get(PMC_OA_BASE, params=params, timeout=30)
        r.raise_for_status()

        # Extract text content from XML tags (simple approach, no heavy XML parser needed)
        import re
        xml = r.text

        # Pull title
        title_match = re.search(r"<article-title>(.*?)</article-title>", xml, re.DOTALL)
        title = title_match.group(1).strip() if title_match else f"PMC{pmc_id}"
        title = re.sub(r"<[^>]+>", "", title)  # strip inner tags

        # Pull abstract
        abstract_match = re.search(r"<abstract>(.*?)</abstract>", xml, re.DOTALL)
        abstract = abstract_match.group(1) if abstract_match else ""
        abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

        # Pull body paragraphs
        body_matches = re.findall(r"<p>(.*?)</p>", xml, re.DOTALL)
        body = " ".join(body_matches)
        body = re.sub(r"<[^>]+>", " ", body).strip()

        if not abstract and not body:
            return None

        return f"Title: {title}\n\nAbstract:\n{abstract}\n\nBody:\n{body[:8000]}"
    except Exception as e:
        print(f"  Warning: could not fetch full text for PMC{pmc_id}: {e}")
        return None


def fetch_abstract_fallback(pmc_id: str) -> str | None:
    """Fallback: fetch abstract via eFetch if full text is not open access."""
    try:
        params = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "abstract",
            "retmode": "text",
        }
        if API_KEY:
            params["api_key"] = API_KEY
        r = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=20)
        r.raise_for_status()
        text = r.text.strip()
        return text if len(text) > 100 else None
    except Exception:
        return None


def fetch_and_save(pmc_ids: list[str], outdir: Path):
    """Download articles and save as .txt files."""
    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    delay = 0.11 if API_KEY else 0.34  # respect NCBI rate limits

    for i, pmc_id in enumerate(pmc_ids, 1):
        out_path = outdir / f"PMC{pmc_id}.txt"
        if out_path.exists():
            print(f"[{i}/{len(pmc_ids)}] PMC{pmc_id} already exists, skipping.")
            skipped += 1
            continue

        print(f"[{i}/{len(pmc_ids)}] Fetching PMC{pmc_id}...", end=" ")
        text = fetch_full_text(pmc_id)

        if not text:
            print("full text unavailable, trying abstract fallback...", end=" ")
            text = fetch_abstract_fallback(pmc_id)

        if text:
            out_path.write_text(text, encoding="utf-8")
            print(f"saved ({len(text)} chars)")
            saved += 1
        else:
            print("no content, skipping.")
            skipped += 1

        time.sleep(delay)

    print(f"\nDone. Saved: {saved} | Skipped/unavailable: {skipped}")
    print(f"Papers saved to: {outdir.resolve()}")
    print(f"\nNext step — ingest into Pinecone:\n  python rag/ingest.py --dir {outdir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PMC papers for RAG ingestion")
    parser.add_argument("--query", required=True, help='Search query e.g. "PTPRN2 ALS"')
    parser.add_argument("--max", type=int, default=50, help="Max number of articles to fetch (default: 50)")
    parser.add_argument("--outdir", default="/app/papers", help="Output directory for text files")
    args = parser.parse_args()

    ids = search_pmc(args.query, args.max)
    if ids:
        fetch_and_save(ids, Path(args.outdir))
    else:
        print("No results found. Try a broader query.")
