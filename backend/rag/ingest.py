"""
RAG ingestion pipeline: load PDFs/text files into Pinecone.
Usage: python rag/ingest.py --dir /path/to/papers
"""
import os
import argparse
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


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

    # Load documents (PDFs and .txt files)
    from langchain_community.document_loaders import DirectoryLoader
    docs = []
    pdf_loader = PyPDFDirectoryLoader(directory)
    docs += pdf_loader.load()
    txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader,
                                  loader_kwargs={"encoding": "utf-8"})
    docs += txt_loader.load()
    print(f"Loaded {len(docs)} documents")

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Embed and upsert
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print(f"Ingested {len(chunks)} chunks into Pinecone index '{index_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing PDF papers")
    args = parser.parse_args()
    ingest(args.dir)
