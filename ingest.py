"""
ingest.py — Embed your personal documents into ChromaDB.

Run this once after setup_db.py, and again whenever you add new documents.

    python ingest.py

Document layout:
    data/documents/facts/     → resume, bio, skills, project descriptions
                                (.txt or .pdf files)
    data/documents/linkedin/  → past LinkedIn posts you've written
                                (.txt files, one post per file)

The ingest is idempotent — it uses upsert with deterministic IDs derived
from file path + chunk index, so re-running won't create duplicates.
"""

import logging
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from config import (
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    CHROMA_DB_PATH, FACTS_DIR, LINKEDIN_DIR,
    CHROMA_FACTS_COLLECTION, CHROMA_LINKEDIN_COLLECTION,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Embedding model ────────────────────────────────────────────────────────
embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


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


def _stable_id(filepath: str, chunk_index: int) -> str:
    """Deterministic chunk ID so re-ingesting the same file doesn't duplicate."""
    raw = f"{filepath}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_documents(directory: Path, extensions: tuple = (".txt", ".pdf")):
    """Load all documents from a directory recursively."""
    docs = []
    if not directory.exists():
        log.warning(f"  Directory not found: {directory}")
        log.warning(f"  Create it and add your documents, then re-run ingest.py")
        return docs

    files = [f for f in directory.rglob("*") if f.suffix in extensions]
    if not files:
        log.warning(f"  No {extensions} files found in {directory}")
        return docs

    for file in files:
        try:
            if file.suffix == ".txt":
                loaded = TextLoader(str(file), encoding="utf-8").load()
            elif file.suffix == ".pdf":
                loaded = PyPDFLoader(str(file)).load()
            else:
                continue
            # Tag each document with its source path for traceability
            for doc in loaded:
                doc.metadata["source_file"] = str(file.relative_to(directory))
            docs.extend(loaded)
            log.info(f"  Loaded: {file.name}")
        except Exception as e:
            log.warning(f"  Failed to load {file.name}: {e}")

    return docs


def ingest_facts():
    """Embed personal fact documents (resume, bio, skills, projects)."""
    log.info("\n── Personal Facts ────────────────────────────────────────")
    docs = _load_documents(FACTS_DIR, extensions=(".txt", ".pdf"))
    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        log.warning("  No chunks produced after splitting.")
        return 0

    texts = [c.page_content for c in chunks]
    ids   = [_stable_id(c.metadata.get("source_file", "unknown"), i) for i, c in enumerate(chunks)]

    log.info(f"  Embedding {len(chunks)} chunks...")
    vectors = embedder.embed_documents(texts)

    client     = _get_client()
    collection = _get_collection(client, CHROMA_FACTS_COLLECTION)
    collection.upsert(ids=ids, documents=texts, embeddings=vectors)

    log.info(f"  ✅ {len(chunks)} fact chunks stored in '{CHROMA_FACTS_COLLECTION}'.")
    return len(chunks)


def ingest_linkedin_examples():
    """Embed past LinkedIn posts for style matching."""
    log.info("\n── LinkedIn Examples ─────────────────────────────────────")
    docs = _load_documents(LINKEDIN_DIR, extensions=(".txt",))
    if not docs:
        return 0

    # LinkedIn posts are short — keep chunks large to preserve full post context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n"],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        log.warning("  No chunks produced after splitting.")
        return 0

    texts = [c.page_content for c in chunks]
    ids   = [_stable_id(c.metadata.get("source_file", "unknown"), i) for i, c in enumerate(chunks)]

    log.info(f"  Embedding {len(chunks)} post chunks...")
    vectors = embedder.embed_documents(texts)

    client     = _get_client()
    collection = _get_collection(client, CHROMA_LINKEDIN_COLLECTION)
    collection.upsert(ids=ids, documents=texts, embeddings=vectors)

    log.info(f"  ✅ {len(chunks)} LinkedIn example chunks stored in '{CHROMA_LINKEDIN_COLLECTION}'.")
    return len(chunks)


def print_stats():
    """Print current ChromaDB collection counts."""
    client = _get_client()
    log.info("\n── ChromaDB Stats ────────────────────────────────────────")
    for name in [CHROMA_FACTS_COLLECTION, CHROMA_LINKEDIN_COLLECTION]:
        try:
            count = client.get_collection(name).count()
            log.info(f"  {name}: {count} chunks")
        except Exception:
            log.info(f"  {name}: (not yet created)")


def main():
    log.info("=" * 54)
    log.info("  Personal Assistant — Document Ingestion")
    log.info("=" * 54)
    log.info(f"\nOllama embed model: {OLLAMA_EMBED_MODEL}")
    log.info(f"ChromaDB path:      {CHROMA_DB_PATH}")

    total = 0
    total += ingest_facts()
    total += ingest_linkedin_examples()

    print_stats()

    log.info("\n" + "=" * 54)
    if total > 0:
        log.info(f"✅ Ingestion complete — {total} total chunks embedded.")
    else:
        log.info("⚠️  Nothing was ingested. Add documents and re-run.")
        log.info(f"   facts/   → {FACTS_DIR}")
        log.info(f"   linkedin/ → {LINKEDIN_DIR}")
    log.info("=" * 54)


if __name__ == "__main__":
    main()
