"""
subgraphs/rag.py — RAG (personal knowledge base) subgraph.

Pipeline: START → researcher_node → search_response_node → END

researcher_node:
  - Fires asyncio.gather for concurrent: LTM store search + ChromaDB facts + optional web
  - Web search only if research_mode is hybrid or open_book (set by router)
  - Keyword extraction from user query drives all three searches

search_response_node:
  - Synthesizes ChromaDB facts + LTM context + optional web into a grounded answer
  - Structured system prompt prevents adding info beyond what sources say
  - If sources are thin, says so and suggests what to add to documents
"""

import asyncio
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

from state import GlobalState
from llm_config import llm_write as llm_fast
from tools.web_search import web_search_async
from tools.rag_engine import retrieve_facts

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Shared keyword extractor
# ══════════════════════════════════════════════════════════════════════════

import re

_KW_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "is","are","was","were","be","been","i","my","me","we","you","your","it","its",
    "what","can","could","would","should","have","has","had","do","did",
    "please","help","need","want","like","tell","show","give","find",
    "search","look","check","list","show","get","fetch","retrieve",
}


def _keywords(text: str, max_kw: int = 6) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    seen, out = set(), []
    for w in cleaned.split():
        if w not in _KW_STOP and len(w) > 2 and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) >= max_kw:
                break
    return out


def _rag_queries(text: str) -> list[str]:
    """Build 1-2 search queries from keywords."""
    kw = _keywords(text)
    if not kw:
        return [text[:80]]
    queries = [" ".join(kw[:4])]
    if len(kw) > 4:
        queries.append(" ".join(kw[4:]))
    return queries


# ══════════════════════════════════════════════════════════════════════════
# Researcher node — concurrent I/O
# ══════════════════════════════════════════════════════════════════════════

async def researcher_node(
    state: GlobalState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Concurrent retrieval: ChromaDB facts + LTM store + optional web search.

    All three fire at the same time via asyncio.gather.
    Web search only runs for hybrid/open_book research_mode.
    """
    messages   = state.get("messages", [])
    from langchain_core.messages import HumanMessage as _HM
    human_msgs = [m for m in messages if isinstance(m, _HM)]
    user_input = human_msgs[-1].content if human_msgs else ""
    research_mode  = state.get("research_mode", "closed_book")
    search_queries = state.get("search_queries", [])  # from router
    user_id        = state.get("user_id", "default")
    ltm_context    = state.get("ltm_context", "")     # already loaded by memory_inject

    queries = search_queries or _rag_queries(user_input)
    kw      = _keywords(user_input)

    # ── Async wrappers for sync libs ───────────────────────────────────────
    loop = asyncio.get_running_loop()

    async def _chroma_facts() -> str:
        return await loop.run_in_executor(None, retrieve_facts, queries)

    async def _web() -> str:
        if research_mode in ("hybrid", "open_book") and queries:
            return await web_search_async(queries[:2], max_results=4)
        return ""

    # ── Fire concurrently ──────────────────────────────────────────────────
    tasks = [_chroma_facts(), _web()]
    chroma_result, web_result = await asyncio.gather(*tasks, return_exceptions=True)

    chroma_facts = chroma_result if not isinstance(chroma_result, Exception) else ""
    web_results  = web_result    if not isinstance(web_result,    Exception) else ""

    if isinstance(chroma_result, Exception):
        logger.warning(f"[rag researcher] ChromaDB failed: {chroma_result}")
    if isinstance(web_result, Exception):
        logger.warning(f"[rag researcher] Web search failed: {web_result}")

    logger.debug(
        f"[rag researcher] chroma={'yes' if chroma_facts else 'empty'} "
        f"web={'yes' if web_results else 'skip'} "
        f"ltm={'yes' if ltm_context else 'empty'}"
    )

    # Store results in state for search_response_node
    # Using a convention: store in messages metadata or pass via state fields.
    # We use a temporary state update pattern — these go into the subgraph's
    # local state and are NOT stored in the PostgreSQL checkpoint.
    return {
        "_rag_chroma_facts": chroma_facts,
        "_rag_web_results":  web_results,
        "current_agent":     "Researcher",
    }


# ══════════════════════════════════════════════════════════════════════════
# Search response node
# ══════════════════════════════════════════════════════════════════════════

_SEARCH_RESPONSE_SYSTEM = """\
You are answering a question using the user's personal knowledge base.

RULES:
- Synthesize ONLY what the provided sources say.
- NEVER add facts not present in the sources.
- If sources are thin or empty, say clearly:
  "I couldn't find much about this in your profile."
  Then suggest what the user should add to their documents.
- Cite sources naturally: "According to your resume...", "Your documents mention..."
- Be direct and concise. No filler.

SOURCE PRIORITY: Documents > LTM Memory > Web (web is supplementary context only)\
"""


async def search_response_node(state: GlobalState) -> dict:
    """
    Synthesize all retrieved sources into a grounded answer.
    """
    from config import STM_WINDOW_SIZE
    messages     = state.get("messages", [])
    from langchain_core.messages import HumanMessage as _HM
    user_input   = next((m.content for m in reversed(messages) if isinstance(m, _HM)), "")
    ltm_context  = state.get("ltm_context", "")
    chroma_facts = state.get("_rag_chroma_facts", "")
    web_results  = state.get("_rag_web_results", "")

    # Build source block
    parts = []
    if chroma_facts and chroma_facts.strip():
        parts.append(f"[FROM YOUR DOCUMENTS]:\n{chroma_facts}")
    if ltm_context and ltm_context.strip():
        parts.append(f"[FROM MEMORY]:\n{ltm_context}")
    if web_results and web_results.strip() and "no results" not in web_results.lower():
        parts.append(f"[FROM WEB (supplementary)]:\n{web_results}")

    if not parts:
        source_block = (
            "No relevant information found in your personal database.\n\n"
            "To improve results:\n"
            "  1. Add documents to data/documents/facts/ (resume, bio, skills)\n"
            "  2. Run: python ingest.py\n"
            "  3. Continue chatting — facts are extracted automatically over time."
        )
    else:
        source_block = "\n\n".join(parts)

    try:
        response = await llm_fast.ainvoke([
            SystemMessage(content=_SEARCH_RESPONSE_SYSTEM),
            HumanMessage(content=f"Question: {user_input}\n\nSources:\n{source_block}\n\nAnswer:"),
        ])
        reply = response.content.strip()
    except Exception as e:
        logger.error(f"[rag search_response] LLM failed: {e}")
        reply = f"Error generating response: {e}"

    return {
        "messages":      [AIMessage(content=reply)],
        "current_agent": "Database Search",
    }


# ══════════════════════════════════════════════════════════════════════════
# Subgraph assembly
# ══════════════════════════════════════════════════════════════════════════

_g = StateGraph(GlobalState)
_g.add_node("researcher",      researcher_node)
_g.add_node("search_response", search_response_node)

_g.add_edge(START,             "researcher")
_g.add_edge("researcher",      "search_response")
_g.add_edge("search_response", END)

rag_subgraph = _g.compile()
