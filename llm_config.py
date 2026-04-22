"""
llm_config.py — Three LLM tier instances.

Tier        Model           Use                         Temp    num_predict
────────────────────────────────────────────────────────────────────────────
llm_write   llama3.1:8b     Generator, style_matcher    0.65    700
llm_precise qwen2.5:7b      Evaluator, clarifier,       0.0     500
                             LTM extractor
llm_router  llama3.2:3b     Router LLM fallback         0.0     200

Why three models instead of one?
  - llm_write   needs creativity. llama3.1:8b produces far better prose
                than a 3b model at the same temperature.
  - llm_precise needs reliability. temp=0 + structured output = no
                hallucinated JSON. A smaller model is fine here because
                the task is classification/extraction, not generation.
  - llm_router  only runs when the regex fast-path misses. It must be
                as fast as possible — 3b model keeps this under 300ms.

num_predict caps token generation per call. Smaller cap = faster response.
A router classification never needs 500 tokens. A LinkedIn post might need 700.
Setting this correctly is one of the best local latency levers you have.

num_ctx sets the context window. We keep it tight to avoid OOM on smaller GPUs.
The STM summarization + recent messages strategy means we never need a huge
context window for inference — the heavy lifting is done by summarization.
"""

from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL_WRITE, OLLAMA_MODEL_PRECISE, OLLAMA_MODEL_ROUTER


# ── Creative writing ── generator, style_matcher ───────────────────────────
llm_write = ChatOllama(
    model=OLLAMA_MODEL_WRITE,
    temperature=0.65,
    base_url=OLLAMA_BASE_URL,
    num_ctx=4096,
    num_predict=700,
)

# ── Structured output ── evaluator, clarifier, LTM extractor ──────────────
llm_precise = ChatOllama(
    model=OLLAMA_MODEL_PRECISE,
    temperature=0.0,
    base_url=OLLAMA_BASE_URL,
    num_ctx=3072,
    num_predict=500,
)

# ── Fast routing classification ── router LLM fallback only ───────────────
llm_router = ChatOllama(
    model=OLLAMA_MODEL_ROUTER,
    temperature=0.0,
    base_url=OLLAMA_BASE_URL,
    num_ctx=1024,
    num_predict=200,
)

# ── Summarization (STM) ────────────────────────────────────────────────────
# Reuses llm_precise — structured, deterministic, short output
llm_summarizer = ChatOllama(
    model=OLLAMA_MODEL_PRECISE,
    temperature=0.0,
    base_url=OLLAMA_BASE_URL,
    num_ctx=2048,
    num_predict=350,
)
