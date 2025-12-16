# Genomics AI Research Agent

An AI-powered research assistant for biomedical researchers to analyze genes and diseases. Built with LangChain, GPT-4o, Pinecone RAG, and public genomics databases — all running via Docker.

## Features

**Literature Retrieval (RAG)**
- Fetches full-text papers from PubMed Central (PMC) using the NCBI API
- Ingests PDFs and text files into a Pinecone vector database
- Semantic search over your literature corpus on every query

**Genomics Database Queries**
- Ensembl REST API — gene structure, transcripts, biotype, chromosomal location
- NCBI Gene — gene summaries, functional annotations, map location
- GWAS Catalog — disease-gene associations and significant SNPs

**Splicing Analysis**
- Fast splicing prediction via SpliceAI-lookup public API (no setup required)
- Deep-learning splicing prediction via local Spliformer model with attention weight score (AWS) heatmaps
- Spliformer motif mode visualizes which splicing motifs are affected in wildtype vs variant sequences

**Pathway Enrichment**
- GO Biological Process and KEGG pathway enrichment via GSEApy

**Reports**
- Auto-generated HTML reports per analysis
- Reports served at `http://localhost:8000/reports/` or uploaded to S3 in cloud deployments

**Chat UI**
- Dark-themed web interface at `http://localhost:3000`
- Markdown rendering with inline heatmap display and auto-download
- Conversation history maintained across turns


## Setup

```bash
cp .env.example .env
# Fill in your API keys in .env
```

Required keys:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `NCBI_API_KEY` (free at https://www.ncbi.nlm.nih.gov/account/)

## Run

```bash
docker compose up --build
```

- Chat UI: http://localhost:3000
- API docs: http://localhost:8000/docs
- Reports: http://localhost:8000/reports/

## Literature Ingestion (RAG)

**Step 1 — Fetch papers from PubMed Central:**
```bash
docker compose exec backend python rag/fetch_papers.py --query "PTPRN2 ALS" --max 100 --outdir /app/papers
```

Run multiple queries to build a richer index:
```bash
docker compose exec backend python rag/fetch_papers.py --query "ALS motor neuron disease genetics" --max 100 --outdir /app/papers
docker compose exec backend python rag/fetch_papers.py --query "TDP-43 neurodegeneration" --max 100 --outdir /app/papers
```

**Step 2 — Ingest into Pinecone:**
```bash
docker compose exec backend python rag/ingest.py --dir /app/papers
```

Supports both PDF and `.txt` files. Already-ingested chunks are skipped automatically.

## Spliformer

Spliformer runs as a sidecar container and downloads the reference genome on first start (~3GB). The API is unavailable during the download.

Supported assemblies: `hg38` (default), `hg19`

Variant format: `chr:pos:ref:alt` — e.g. `chr2:179642185:G:A`

Example queries:
- `Predict splicing for chr2:179642185:G:A using Spliformer hg19`
- `Show Spliformer motif heatmaps for chr2:179642185:G:A hg19`

Heatmap images are automatically downloaded by the frontend when returned by the agent.

## Example Queries

- `Analyze PTPRN2 in ALS`
- `What GWAS variants are associated with ALS?`
- `Run enrichment analysis on SOD1, FUS, TDP43, C9orf72, PTPRN2`
- `Predict splicing impact for variant 7:117548628:A:G`
- `Get Ensembl annotation for PTPRN2`
- `Generate a report for PTPRN2 in ALS`

## Splicing Score Interpretation

| Max delta score | Impact |
|---|---|
| ≥ 0.5 | HIGH |
| 0.2 – 0.5 | MODERATE |
| < 0.2 | LOW |

Scores: DS_AG (acceptor gain), DS_AL (acceptor loss), DS_DG (donor gain), DS_DL (donor loss)
