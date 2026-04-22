"""
config.py — Single source of truth for all configuration.

Every value comes from .env first, with a sensible local default as fallback.
Nothing is hardcoded outside this file. All other modules import from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root, override any stale shell env vars
load_dotenv(Path(__file__).parent / ".env", override=True)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
DOCUMENTS_DIR  = DATA_DIR / "documents"
FACTS_DIR      = DOCUMENTS_DIR / "facts"
LINKEDIN_DIR   = DOCUMENTS_DIR / "linkedin"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")
OLLAMA_MODEL_WRITE   = os.getenv("OLLAMA_MODEL_WRITE",   "llama3.1:8b")   # creative writing
OLLAMA_MODEL_PRECISE = os.getenv("OLLAMA_MODEL_PRECISE", "qwen2.5:7b")    # structured JSON
OLLAMA_MODEL_ROUTER  = os.getenv("OLLAMA_MODEL_ROUTER",  "llama3.2:3b")   # fast classification
OLLAMA_EMBED_MODEL   = os.getenv("OLLAMA_EMBED_MODEL",   "nomic-embed-text")

# ── PostgreSQL ─────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
PG_PORT     = os.getenv("POSTGRES_PORT",     "5442")
PG_DB       = os.getenv("POSTGRES_DB",       "postgres")
PG_USER     = os.getenv("POSTGRES_USER",     "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

PG_CONN_STRING = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# ── STM ────────────────────────────────────────────────────────────────────
# Keep this many recent messages raw in the checkpoint (immediate context)
STM_WINDOW_SIZE = int(os.getenv("STM_WINDOW_SIZE", "6"))

# Trigger summarization when total message count exceeds this
STM_SUMMARIZE_AFTER = int(os.getenv("STM_SUMMARIZE_AFTER", "10"))

# ── LTM ────────────────────────────────────────────────────────────────────
LTM_MAX_FACTS_PER_TURN    = int(os.getenv("LTM_MAX_FACTS_PER_TURN",    "8"))
LTM_RETRIEVAL_LIMIT       = int(os.getenv("LTM_RETRIEVAL_LIMIT",       "8"))
LTM_SIMILARITY_THRESHOLD  = float(os.getenv("LTM_SIMILARITY_THRESHOLD","0.60"))
LTM_DEDUP_THRESHOLD       = float(os.getenv("LTM_DEDUP_THRESHOLD",     "0.92"))

# LTM store namespaces — (namespace_tuple, key)
# User facts live under ("ltm", user_id) so different users are isolated
LTM_NAMESPACE_PREFIX = "ltm"

# ── Router ─────────────────────────────────────────────────────────────────
# LLM router only fires when regex fast-path returns None.
# Below this confidence the LLM result is ignored and we fall back to "chat".
ROUTER_MIN_CONFIDENCE = float(os.getenv("ROUTER_MIN_CONFIDENCE", "0.70"))

# ── Web Search ─────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS      = int(os.getenv("WEB_SEARCH_MAX_RESULTS",      "5"))
WEB_SEARCH_SCRAPE_TOP_N     = int(os.getenv("WEB_SEARCH_SCRAPE_TOP_N",     "2"))
WEB_OPEN_BOOK_RECENCY_DAYS  = int(os.getenv("WEB_OPEN_BOOK_RECENCY_DAYS",  "7"))
WEB_HYBRID_RECENCY_DAYS     = int(os.getenv("WEB_HYBRID_RECENCY_DAYS",     "45"))

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_FACTS_COLLECTION    = "personal_facts"
CHROMA_LINKEDIN_COLLECTION = "linkedin_examples"

# ── Server ─────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
