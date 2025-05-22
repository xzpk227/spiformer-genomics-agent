import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from pinecone import Pinecone


def get_vectorstore():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(
        index=pc.Index(os.environ["PINECONE_INDEX_NAME"]),
        embedding=embeddings,
    )


@tool
def search_literature(query: str) -> str:
    """Search scientific literature in the vector database for a given query about genes or diseases."""
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=5)
        if not docs:
            return "No relevant literature found for this query."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", "Untitled")
            results.append(f"[{i}] {title} ({source})\n{doc.page_content[:500]}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Literature search error: {str(e)}"
