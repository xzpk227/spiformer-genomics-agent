"""
RAG ingestion pipeline: load PDFs/text files into Pinecone.
Skips chunks already present in the index using a content hash as the vector ID.
Usage: python rag/ingest.py --dir /path/to/papers
"""
import os
import hashlib
import argparse
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def chunk_id(text: str) -> str:
    """Stable ID from content hash — same chunk always gets the same ID."""
    return hashlib.sha256(text.encode()).hexdigest()[:48]


def ingest(directory: str):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]

    # Create index if it doesn't exist
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {index_name}")

    index = pc.Index(index_name)

    # Load documents
    docs = []
    docs += PyPDFDirectoryLoader(directory).load()
    docs += DirectoryLoader(
        directory, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    ).load()
    print(f"Loaded {len(docs)} documents")

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Assign stable IDs based on content hash
    ids = [chunk_id(c.page_content) for c in chunks]

    # Check which IDs already exist in Pinecone (batch fetch in groups of 100)
    existing_ids = set()
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        response = index.fetch(ids=batch)
        existing_ids.update(response.vectors.keys())

    # Filter to only new chunks
    new_chunks = [(chunk, id_) for chunk, id_ in zip(chunks, ids) if id_ not in existing_ids]
    print(f"Skipping {len(existing_ids)} already-ingested chunks")
    print(f"Ingesting {len(new_chunks)} new chunks")

    if not new_chunks:
        print("Nothing new to ingest.")
        return

    # Embed and upsert only new chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = PineconeVectorStore(index=index, embedding=embeddings)
    new_docs, new_ids = zip(*new_chunks)

    # Extract title per source file and propagate to all chunks from that file
    import re
    source_titles: dict[str, str] = {}
    docs_to_add = list(new_docs)

    # First pass: find the title for each source
    for doc in docs_to_add:
        src = doc.metadata.get("source", "")
        if src not in source_titles:
            m = re.search(r"^Title:\s*(.+)", doc.page_content, re.IGNORECASE | re.MULTILINE)
            if m:
                source_titles[src] = m.group(1).strip()[:120]

    # Second pass: stamp title on every chunk from that source
    for doc in docs_to_add:
        src = doc.metadata.get("source", "")
        doc.metadata["title"] = source_titles.get(src, "")

    vs.add_documents(docs_to_add, ids=list(new_ids))
    print(f"Done. {len(new_chunks)} new chunks ingested into '{index_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing PDF/txt papers")
    args = parser.parse_args()
    ingest(args.dir)
