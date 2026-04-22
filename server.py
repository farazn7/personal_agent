"""
server.py — FastAPI backend with SSE streaming.

HITL FLOW (state-flag, no interrupt()):
  1. POST /api/chat  → graph runs → if _li_needs_hitl: emit hitl SSE
  2. POST /api/hitl  → read checkpoint, build resume state, re-run graph
     Multi-turn: clarifier re-evaluates with accumulated answers each round.

subgraphs=True in astream() is SAFE here because we no longer use
interrupt() — that was the only function that needed the config contextvar.
subgraphs=True surfaces events from inside linkedin/rag subgraphs so the
UI sees Researcher, Clarifier, Generator etc. rather than a black box.
"""

import json
import uuid
import logging
import contextlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_agent_graph = None
DEFAULT_USER = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent_graph
    logger.info("Starting up — building graph...")
    from graph import get_graph_async
    _agent_graph = await get_graph_async()
    logger.info("Server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Personal Assistant API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _build_initial_state(message: str) -> dict:
    from langchain_core.messages import HumanMessage
    return {
        "user_id":            DEFAULT_USER,
        "messages":           [HumanMessage(content=message)],
        "stm_summary":        "",
        "route":              "",
        "research_mode":      "",
        "search_queries":     [],
        "ltm_context":        "",
        "current_agent":      "",
        "_li_web_results":    "",
        "_li_style_examples": "",
        "_li_hitl_answers":   {},
        "_li_draft_v1":       "",
        "_li_draft_v2":       "",
        "_li_final_post":     "",
        "_li_needs_hitl":     False,
        "_li_hitl_questions": [],
        "_li_hitl_complete":  False,
        "_li_hitl_rounds":    0,
        "_rag_chroma_facts":  "",
        "_rag_web_results":   "",
    }


def _agent_detail(agent: str, full_state: dict) -> str:
    if agent == "Memory":
        ltm = full_state.get("ltm_context", "")
        return f"loaded {ltm.count('•')} facts" if ltm else "no prior memory"
    if agent == "Router":
        route = full_state.get("route", "?")
        mode  = full_state.get("research_mode", "")
        nq    = len(full_state.get("search_queries") or [])
        return f"→ {route} · {mode} · {nq} queries" if (mode and mode != "closed_book") else f"→ {route}"
    if agent == "Researcher":
        web  = full_state.get("_li_web_results") or full_state.get("_rag_web_results") or ""
        ltm  = full_state.get("ltm_context", "")
        mode = full_state.get("research_mode", "closed_book")
        parts = []
        if web:
            n = web.count("Source:")
            parts.append(f"web: {n} results" if n else "web ✓")
        elif mode in ("hybrid", "open_book"):
            parts.append("web: no results")
        if ltm:
            n = ltm.count("•")
            parts.append(f"memory: {n} facts" if n else "memory ✓")
        return " · ".join(parts) if parts else "style examples loaded"
    if agent == "Clarifier":
        rnd = full_state.get("_li_hitl_rounds", 0)
        if full_state.get("_li_hitl_complete") and not full_state.get("_li_needs_hitl"):
            return f"re-evaluated with answers (round {rnd})"
        return "checking what info is needed"
    if agent == "Generator":
        return "writing from fact list…"
    if agent == "Evaluator":
        return "checking for invented content…"
    if agent == "Style Matcher":
        return "matching your voice" if full_state.get("_li_style_examples") else "no past posts — skipping"
    if agent == "Chatbot":
        return ""
    if agent == "Database Search":
        return "querying your documents…"
    return ""


async def _run_stream(graph_input, thread_id: str) -> AsyncGenerator[str, None]:
    """Stream graph execution as SSE events."""
    from langchain_core.messages import AIMessage

    config     = {"configurable": {"thread_id": thread_id}}
    full_state: dict = {}
    last_agent = ""

    # subgraphs=True: surfaces node events from inside compiled subgraphs.
    # Safe now that interrupt() is gone (interrupt needed the config contextvar
    # which subgraphs=True was corrupting).
    astream_gen = _agent_graph.astream(
        graph_input,
        config=config,
        stream_mode="updates",
        subgraphs=True,
    )

    try:
        async for namespace, event in astream_gen:
            for node_name, node_output in event.items():
                if not isinstance(node_output, dict):
                    continue
                full_state.update(node_output)
                agent = node_output.get("current_agent", "")
                if agent and agent != last_agent:
                    last_agent = agent
                    detail = _agent_detail(agent, full_state)
                    yield _sse({"type": "agent", "agent": agent, "detail": detail})

        # State-flag HITL check
        if full_state.get("_li_needs_hitl"):
            questions = full_state.get("_li_hitl_questions", [])
            logger.info(f"[stream] HITL: {len(questions)} questions")
            yield _sse({"type": "hitl", "questions": questions, "thread_id": thread_id})
            yield _sse({"type": "done"})
            return

        # Final response
        route = full_state.get("route", "chat")
        if route == "linkedin":
            post = (
                full_state.get("_li_final_post")
                or full_state.get("_li_draft_v2")
                or full_state.get("_li_draft_v1")
                or ""
            )
            yield _sse({"type": "linkedin", "content": post})
        else:
            ai_msgs = [m for m in full_state.get("messages", []) if isinstance(m, AIMessage)]
            resp = ai_msgs[-1].content if ai_msgs else "Sorry, I couldn't generate a response."
            yield _sse({"type": "chat", "content": resp})

        yield _sse({"type": "done"})

    except Exception as e:
        logger.exception(f"[stream] error: {e}")
        yield _sse({"type": "error", "content": str(e)})
    finally:
        with contextlib.suppress(Exception):
            await astream_gen.aclose()


class ChatRequest(BaseModel):
    message:   str
    thread_id: str = ""


class HitlRequest(BaseModel):
    original_input: str
    answers:        dict
    thread_id:      str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    state     = _build_initial_state(req.message)
    return StreamingResponse(
        _run_stream(state, thread_id),
        media_type="text/event-stream",
        headers={
            "X-Thread-Id":       thread_id,
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/hitl")
async def hitl(req: HitlRequest):
    """
    HITL resume.

    Reads the checkpoint to get ALL state from the first run
    (route, research results, ltm_context, round count, etc.)
    then builds a resume state that:
      1. Passes messages=[] so add_messages appends nothing (no duplicate)
      2. Passes _li_hitl_complete=True so researcher + clarifier skip LLM
      3. Passes the MERGED accumulated answers (existing + new)
      4. Increments _li_hitl_rounds
      5. Resets draft fields so generator writes fresh output

    Multi-turn: after resume, clarifier re-evaluates. If still needs info
    AND rounds < 2, it sets _li_needs_hitl=True again → another hitl SSE.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    # Read checkpoint from first run
    cp: dict = {}
    try:
        snapshot = await _agent_graph.aget_state(config)
        if snapshot and snapshot.values:
            cp = dict(snapshot.values)
    except Exception as e:
        logger.warning(f"[hitl] checkpoint read failed ({e})")

    # Merge answers: existing accumulated + new from this submission
    existing_answers = cp.get("_li_hitl_answers", {}) or {}
    merged_answers   = {**existing_answers, **req.answers}
    new_rounds       = (cp.get("_li_hitl_rounds", 0) or 0) + 1
    logger.info(f"[hitl] round {new_rounds}, {len(merged_answers)} accumulated answers")

    resume_state = {
        "user_id":            cp.get("user_id", DEFAULT_USER),
        "messages":           [],   # add_messages: appends nothing → no duplicate HumanMessage
        "stm_summary":        cp.get("stm_summary", ""),
        "route":              cp.get("route", "linkedin"),
        "research_mode":      cp.get("research_mode", "closed_book"),
        "search_queries":     cp.get("search_queries", []),
        "ltm_context":        cp.get("ltm_context", ""),
        "current_agent":      "",
        "_li_web_results":    cp.get("_li_web_results", ""),
        "_li_style_examples": cp.get("_li_style_examples", ""),
        "_li_hitl_answers":   merged_answers,
        "_li_hitl_complete":  True,
        "_li_needs_hitl":     False,
        "_li_hitl_questions": [],
        "_li_hitl_rounds":    new_rounds,
        "_li_draft_v1":       "",
        "_li_draft_v2":       "",
        "_li_final_post":     "",
        "_rag_chroma_facts":  cp.get("_rag_chroma_facts", ""),
        "_rag_web_results":   cp.get("_rag_web_results", ""),
    }

    return StreamingResponse(
        _run_stream(resume_state, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/session-messages")
async def get_session_messages(thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state  = await _agent_graph.aget_state(config)
        if not state or not state.values:
            return {"messages": []}
        result = []
        for m in state.values.get("messages", []):
            mtype = type(m).__name__.lower()
            if "human" in mtype:
                result.append({"type": "human", "content": m.content})
            elif "ai" in mtype and m.content:
                result.append({"type": "ai", "content": m.content})
        return {"messages": result}
    except Exception as e:
        logger.warning(f"session-messages: {e}")
        return {"messages": []}


@app.get("/api/sessions")
async def list_sessions():
    try:
        import psycopg
        from config import PG_CONN_STRING
        async with await psycopg.AsyncConnection.connect(PG_CONN_STRING, connect_timeout=3) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT thread_id, MAX(checkpoint_id) as latest
                    FROM checkpoints WHERE checkpoint_ns = ''
                    GROUP BY thread_id ORDER BY latest DESC LIMIT 30
                """)
                rows = await cur.fetchall()
        return [{"thread_id": r[0], "checkpoint_id": r[1]} for r in rows]
    except Exception:
        return []


@app.get("/api/ltm")
async def get_ltm():
    try:
        from memory.ltm import get_all_ltm_facts
        store = getattr(_agent_graph, "store", None)
        if not store:
            return {"facts": [], "error": "LTM store not initialized"}
        facts = await get_all_ltm_facts(DEFAULT_USER, store)
        return {"facts": facts}
    except Exception as e:
        return {"facts": [], "error": str(e)}


@app.delete("/api/ltm/{key}")
async def delete_ltm(key: str):
    try:
        from memory.ltm import delete_ltm_fact
        store = getattr(_agent_graph, "store", None)
        ok    = await delete_ltm_fact(DEFAULT_USER, key, store)
        return {"ok": ok}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/stats")
async def get_stats():
    result = {}
    try:
        from tools.rag_engine import get_db_stats
        result["chroma"] = get_db_stats()
    except Exception:
        result["chroma"] = {}
    return result


@app.get("/api/health")
async def health():
    status = {"graph": _agent_graph is not None, "ollama": False}
    try:
        import httpx
        from config import OLLAMA_BASE_URL
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        status["ollama"] = r.status_code == 200
    except Exception:
        pass
    return status


from pathlib import Path
UI_INDEX = Path(__file__).parent / "ui" / "dist" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if UI_INDEX.exists():
        return HTMLResponse(UI_INDEX.read_text())
    return HTMLResponse("<h1>UI not built. Run: cd ui && npm run build</h1>")


@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_spa(path: str):
    if UI_INDEX.exists():
        return HTMLResponse(UI_INDEX.read_text())
    return HTMLResponse("<h1>Not found</h1>", status_code=404)
