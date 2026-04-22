"""
graph.py — Main LangGraph pipeline assembly.

Structure:
    START
      -> memory_inject
      -> router
      -> [chat | linkedin | rag]
      -> memory_update
    END

The linkedin subgraph uses state-flag HITL (not interrupt()).
When _li_needs_hitl=True the subgraph routes itself to END internally.
server.py detects the flag after the graph finishes and emits the hitl SSE.
The /api/hitl endpoint re-runs the full graph with _li_hitl_complete=True,
which causes researcher and clarifier to skip their LLM work and the
writing pipeline to run with the user's answers already in state.
"""

import logging
from langgraph.graph import StateGraph, START, END

from state import GlobalState
from agents.router import router_node, route_decision
from agents.memory_nodes import memory_inject_node, memory_update_node

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None, store=None):
    from subgraphs.chat import chat_subgraph
    from subgraphs.linkedin import linkedin_subgraph
    from subgraphs.rag import rag_subgraph

    g = StateGraph(GlobalState)

    g.add_node("memory_inject",  memory_inject_node)
    g.add_node("router",         router_node)
    g.add_node("chat",           chat_subgraph)
    g.add_node("linkedin",       linkedin_subgraph)
    g.add_node("rag",            rag_subgraph)
    g.add_node("memory_update",  memory_update_node)

    g.add_edge(START, "memory_inject")
    g.add_edge("memory_inject", "router")

    g.add_conditional_edges(
        "router",
        route_decision,
        {
            "chat":     "chat",
            "linkedin": "linkedin",
            "rag":      "rag",
        },
    )

    g.add_edge("chat",     "memory_update")
    g.add_edge("linkedin", "memory_update")
    g.add_edge("rag",      "memory_update")
    g.add_edge("memory_update", END)

    return g.compile(checkpointer=checkpointer, store=store)


async def get_graph_async():
    checkpointer = None
    store        = None

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

    try:
        from langgraph.store.memory import InMemoryStore
        from langchain_ollama import OllamaEmbeddings
        from config import OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL
        embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        store = InMemoryStore(index={"embed": embedder, "dims": 768})
        logger.info("InMemoryStore ready.")
    except Exception as e:
        logger.warning(f"LTM store setup failed ({e}) — LTM disabled.")

    graph = build_graph(checkpointer=checkpointer, store=store)
    logger.info("Main graph compiled and ready.")

    # Populate InMemoryStore from PostgreSQL so facts survive restarts
    if store:
        try:
            from memory.ltm import load_ltm_from_postgres
            await load_ltm_from_postgres("default", store)
        except Exception as e:
            logger.warning(f"LTM load from PostgreSQL failed ({e}) — starting with empty memory")

    return graph


def get_graph_sync_fallback():
    from langgraph.checkpoint.memory import MemorySaver
    return build_graph(checkpointer=MemorySaver(), store=None)
