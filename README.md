# Genomics AI Research Agent

Analyze genes and diseases using LangChain, Pinecone RAG, and public genomics databases.

## Setup

```bash
cp .env.example .env
# Fill in your API keys in .env
```

## Run

```bash
docker-compose up --build
```

- Chat UI: http://localhost:3000
- API docs: http://localhost:8000/docs
- Reports: http://localhost:8000/reports/

## Ingest Literature (RAG)

```bash
docker-compose exec backend python rag/ingest.py --dir /path/to/pdfs
```

## Example Queries

- "Analyze PTPRN2 in ALS"
- "Run enrichment analysis on SOD1, FUS, TDP43, C9orf72"
- "What GWAS variants are associated with ALS?"
- "Predict splicing impact for variant 7:117548628:A:G"
