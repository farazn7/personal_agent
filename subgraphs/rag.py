"""
subgraphs/rag.py — RAG (personal knowledge base) subgraph (isolated).

Pipeline: START → researcher_node → synthesizer_node → END
State:   RAGState (4 overlapping + 2 private fields)

researcher_node:
  - Fires asyncio.gather for concurrent: ChromaDB facts + optional web search
  - Web search only if research_mode is hybrid or open_book (set by router)
  - Keyword extraction from user query drives searches

synthesizer_node:
  - Synthesizes ChromaDB facts + LTM context + optional web into a grounded answer
  - Structured system prompt prevents adding info beyond what sources say
  - If sources are thin, says so and suggests what to add to documents
"""

import asyncio
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from states.rag import RAGState
from shared import last_human_content, extract_keywords, build_rag_queries
from llm_config import llm_write as llm_fast
from tools.web_search import web_search_async
from tools.rag_engine import retrieve_facts

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Researcher node — concurrent I/O
# ══════════════════════════════════════════════════════════════════════════

async def researcher_node(state: RAGState) -> dict:
    """
    Concurrent retrieval: ChromaDB facts + optional web search.

    Both fire at the same time via asyncio.gather.
    Web search only runs for hybrid/open_book research_mode.
    """
    messages       = state.get("messages", [])
    user_input     = last_human_content(messages)
    research_mode  = state.get("research_mode", "closed_book")
    search_queries = state.get("search_queries", [])  # from router
    ltm_context    = state.get("ltm_context", "")     # already loaded by memory_inject

    queries = search_queries or build_rag_queries(user_input)

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

    return {
        "chroma_results": chroma_facts,
        "web_results":    web_results,
    }


# ══════════════════════════════════════════════════════════════════════════
# Synthesizer node
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


async def synthesizer_node(state: RAGState) -> dict:
    """
    Synthesize all retrieved sources into a grounded answer.
    """
    messages       = state.get("messages", [])
    user_input     = last_human_content(messages)
    ltm_context    = state.get("ltm_context", "")
    chroma_facts   = state.get("chroma_results", "")
    web_results    = state.get("web_results", "")

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
        logger.error(f"[rag synthesizer] LLM failed: {e}")
        reply = f"Error generating response: {e}"

    return {
        "messages": [AIMessage(content=reply)],
    }


# ══════════════════════════════════════════════════════════════════════════
# Subgraph factory
# ══════════════════════════════════════════════════════════════════════════

def build_rag_subgraph():
    """Build and compile the RAG subgraph. Call this instead of using a module-level singleton."""
    g = StateGraph(RAGState)
    g.add_node("researcher",   researcher_node)
    g.add_node("synthesizer",  synthesizer_node)
    g.add_edge(START,          "researcher")
    g.add_edge("researcher",   "synthesizer")
    g.add_edge("synthesizer",  END)
    return g.compile()
