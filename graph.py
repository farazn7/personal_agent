"""
graph.py — Parent orchestrator graph assembly.

Structure:
  START → memory_inject → router → [chat | rag | linkedin] → memory_update → END

Each subgraph is compiled via its factory function and added as a node.
LangGraph automatically maps overlapping fields between OrchestratorState
and each subgraph's private state schema.

HITL: The LinkedIn subgraph uses interrupt() inside its clarifier node.
When paused, the checkpoint is saved by the parent's checkpointer.
Resume with Command(resume=answers) — the graph continues mid-node.
No re-running memory_inject, router, or researcher.
"""

import logging
from langgraph.graph import StateGraph, START, END

from states.orchestrator import OrchestratorState
from nodes.router import router_node, route_decision
from nodes.memory import memory_inject_node, memory_update_node

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None, store=None):
    """
    Build and compile the parent orchestrator graph.

    Args:
        checkpointer: LangGraph checkpointer (required for interrupt() to work).
        store: LangGraph BaseStore for LTM (InMemoryStore in production).
    """
    from subgraphs.chat import build_chat_subgraph
    from subgraphs.rag import build_rag_subgraph
    from subgraphs.linkedin import build_linkedin_subgraph

    chat_sg     = build_chat_subgraph()
    rag_sg      = build_rag_subgraph()
    linkedin_sg = build_linkedin_subgraph()

    g = StateGraph(OrchestratorState)

    # Nodes
    g.add_node("memory_inject",  memory_inject_node)
    g.add_node("router",         router_node)
    g.add_node("chat",           chat_sg)         # compiled subgraph as node
    g.add_node("rag",            rag_sg)          # compiled subgraph as node
    g.add_node("linkedin",       linkedin_sg)     # compiled subgraph as node
    g.add_node("memory_update",  memory_update_node)

    # Edges
    g.add_edge(START, "memory_inject")
    g.add_edge("memory_inject", "router")

    g.add_conditional_edges(
        "router",
        route_decision,
        {"chat": "chat", "rag": "rag", "linkedin": "linkedin"},
    )

    g.add_edge("chat",     "memory_update")
    g.add_edge("rag",      "memory_update")
    g.add_edge("linkedin", "memory_update")
    g.add_edge("memory_update", END)

    # Compile with checkpointer (REQUIRED for interrupt) and store (for LTM)
    return g.compile(checkpointer=checkpointer, store=store)


async def get_graph_async():
    """
    Production graph with PostgreSQL checkpointer + InMemoryStore for LTM.

    Falls back to MemorySaver if PostgreSQL is unavailable.
    Falls back to no LTM if the embedding model is unavailable.
    """
    checkpointer = None
    store        = None

    # ── Checkpointer: PostgreSQL (preferred) or MemorySaver (fallback) ─────
    try:
        import psycopg
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from config import PG_CONN_STRING
        conn = await psycopg.AsyncConnection.connect(PG_CONN_STRING, autocommit=True)
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()
        logger.info("AsyncPostgresSaver connected.")
    except Exception as e:
        logger.warning(f"PostgresSaver failed ({e}) — falling back to MemorySaver.")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    # ── Store: InMemoryStore with embeddings for LTM ───────────────────────
    try:
        from langgraph.store.memory import InMemoryStore
        from langchain_ollama import OllamaEmbeddings
        from config import OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL
        embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        store = InMemoryStore(index={"embed": embedder, "dims": 768})
        logger.info("InMemoryStore ready.")
    except Exception as e:
        logger.warning(f"LTM store setup failed ({e}) — LTM disabled.")

    # ── Compile ────────────────────────────────────────────────────────────
    graph = build_graph(checkpointer=checkpointer, store=store)
    logger.info("Main graph compiled and ready.")

    # ── Populate InMemoryStore from PostgreSQL so facts survive restarts ───
    if store:
        try:
            from memory.ltm import load_ltm_from_postgres
            await load_ltm_from_postgres("default", store)
        except Exception as e:
            logger.warning(f"LTM load from PostgreSQL failed ({e}) — starting with empty memory")

    return graph


def get_graph_sync_fallback():
    """Synchronous fallback for testing — MemorySaver, no LTM store."""
    from langgraph.checkpoint.memory import MemorySaver
    return build_graph(checkpointer=MemorySaver(), store=None)
