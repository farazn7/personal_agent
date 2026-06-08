"""
tools/rag_engine.py — ChromaDB retrieval for personal documents.

Two collections:
  personal_facts    — resume, bio, skills, project descriptions (ingested by ingest.py)
  linkedin_examples — past LinkedIn posts for style matching

Retrieval functions are synchronous (ChromaDB is sync).
Call them via loop.run_in_executor() from async contexts.
"""

import logging
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langsmith import traceable

from config import (
    CHROMA_DB_PATH, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL,
    CHROMA_FACTS_COLLECTION, CHROMA_LINKEDIN_COLLECTION,
)

logger = logging.getLogger(__name__)

# Singleton embedder — one instance per process
_embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.PersistentClient, name: str):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


@traceable(run_type="retriever", name="chromadb_facts")
def retrieve_facts(queries: List[str], n_results: int = 5) -> str:
    """
    Retrieve personal fact chunks from ChromaDB.
    Deduplicates across multiple queries.
    Returns formatted string or empty-state message.
    """
    try:
        client     = _get_client()
        collection = _get_collection(client, CHROMA_FACTS_COLLECTION)
        count      = collection.count()
        if count == 0:
            return ""

        seen, docs = set(), []
        for query in queries:
            results = collection.query(
                query_embeddings=[_embedder.embed_query(query)],
                n_results=min(n_results, count),
            )
            for doc in results.get("documents", [[]])[0]:
                if doc not in seen:
                    seen.add(doc)
                    docs.append(doc)

        if not docs:
            return ""
        return "\n\n---\n\n".join(docs)

    except Exception as e:
        logger.error(f"[rag] retrieve_facts failed: {e}")
        return ""


@traceable(run_type="retriever", name="chromadb_linkedin_examples")
def retrieve_linkedin_examples(queries: List[str], n_results: int = 3) -> str:
    """
    Retrieve past LinkedIn post examples for style matching.
    Returns formatted string with separator between examples.
    """
    try:
        client     = _get_client()
        collection = _get_collection(client, CHROMA_LINKEDIN_COLLECTION)
        count      = collection.count()
        if count == 0:
            return ""

        seen, docs = set(), []
        for query in queries:
            results = collection.query(
                query_embeddings=[_embedder.embed_query(query)],
                n_results=min(n_results, count),
            )
            for doc in results.get("documents", [[]])[0]:
                if doc not in seen:
                    seen.add(doc)
                    docs.append(doc)

        if not docs:
            return ""
        return "\n\n===== EXAMPLE POST =====\n\n".join(docs)

    except Exception as e:
        logger.error(f"[rag] retrieve_linkedin_examples failed: {e}")
        return ""


def get_db_stats() -> dict:
    """Return document counts per collection (for UI status panel)."""
    client = _get_client()
    stats  = {}
    for name in [CHROMA_FACTS_COLLECTION, CHROMA_LINKEDIN_COLLECTION]:
        try:
            stats[name] = client.get_collection(name).count()
        except Exception:
            stats[name] = 0
    return stats
