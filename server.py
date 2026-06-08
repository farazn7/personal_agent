"""
server.py — FastAPI backend with SSE streaming.

HITL FLOW (interrupt-based):
  1. POST /api/chat  → graph runs → if interrupt() fires: emit hitl SSE
  2. POST /api/hitl  → Command(resume=answers) → graph resumes mid-node
     No re-running router, memory_inject, or researcher. Zero state reconstruction.
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
from langgraph.types import Command

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
    """
    Minimal initial state — only what the parent graph needs to start.
    The router fills route/research_mode/search_queries.
    Memory fills ltm_context/stm_summary.
    Subgraph fields are private and don't exist here.
    """
    from langchain_core.messages import HumanMessage
    return {
        "user_id":  DEFAULT_USER,
        "messages": [HumanMessage(content=message)],
    }


# =============================================================================
# Node name → friendly display name for the UI
# =============================================================================

_NODE_DISPLAY = {
    "memory_inject": "Memory",
    "router":        "Router",
    "chatbot":       "Chatbot",
    "researcher":    "Researcher",
    "clarifier":     "Clarifier",
    "generator":     "Generator",
    "evaluator":     "Evaluator",
    "style_matcher": "Style Matcher",
    "synthesizer":   "Synthesizer",
    "memory_update": "Memory",
}


def _agent_detail(node_name: str, full_state: dict) -> str:
    """Generate a short detail string for the UI based on which node just ran."""
    if node_name == "memory_inject":
        ltm = full_state.get("ltm_context", "")
        return f"loaded {ltm.count('•')} facts" if ltm else "no prior memory"
    if node_name == "router":
        route = full_state.get("route", "?")
        mode  = full_state.get("research_mode", "")
        nq    = len(full_state.get("search_queries") or [])
        return f"→ {route} · {mode} · {nq} queries" if (mode and mode != "closed_book") else f"→ {route}"
    if node_name == "researcher":
        mode = full_state.get("research_mode", "closed_book")
        return f"mode={mode}"
    if node_name == "clarifier":
        return "checking what info is needed"
    if node_name == "generator":
        return "writing from fact list…"
    if node_name == "evaluator":
        return "checking for invented content…"
    if node_name == "style_matcher":
        return "matching your voice"
    return ""


# =============================================================================
# Streaming
# =============================================================================

async def _run_stream(graph_input, thread_id: str) -> AsyncGenerator[str, None]:
    """
    Stream graph execution as SSE events.

    After streaming completes, checks state.next to detect if an interrupt
    occurred (HITL). If so, extracts the interrupt value and yields an
    hitl SSE event. Otherwise, yields the final response.
    """
    from langchain_core.messages import AIMessage

    config     = {"configurable": {"thread_id": thread_id}}
    full_state: dict = {}
    last_agent = ""

    astream_gen = _agent_graph.astream(
        graph_input,
        config=config,
        stream_mode="updates",
    )

    try:
        async for event in astream_gen:
            for node_name, node_output in event.items():
                if not isinstance(node_output, dict):
                    continue
                full_state.update(node_output)

                # Map node name to display name for the UI
                display = _NODE_DISPLAY.get(node_name, node_name)
                if display and display != last_agent:
                    last_agent = display
                    detail = _agent_detail(node_name, full_state)
                    yield _sse({"type": "agent", "agent": display, "detail": detail})

    except Exception as e:
        logger.exception(f"[stream] error: {e}")
        yield _sse({"type": "error", "content": str(e)})
        return
    finally:
        with contextlib.suppress(Exception):
            await astream_gen.aclose()

    # ── Check if we hit an interrupt (HITL) ────────────────────────────────
    try:
        state = await _agent_graph.aget_state(config)
    except Exception as e:
        logger.error(f"[stream] aget_state failed: {e}")
        yield _sse({"type": "error", "content": str(e)})
        return

    if state.next:  # pending nodes = graph was interrupted
        # Extract the interrupt value (the questions dict)
        try:
            interrupt_value = state.tasks[0].interrupts[0].value
            questions = interrupt_value.get("questions", [])
            round_num = interrupt_value.get("round", 1)
            logger.info(f"[stream] HITL interrupt: round {round_num}, {len(questions)} questions")
            yield _sse({
                "type":      "hitl",
                "questions": questions,
                "round":     round_num,
                "thread_id": thread_id,
            })
        except (IndexError, AttributeError) as e:
            logger.error(f"[stream] failed to extract interrupt value: {e}")
            yield _sse({"type": "error", "content": f"HITL interrupt extraction failed: {e}"})
        yield _sse({"type": "done"})
        return

    # ── Graph completed normally — extract final response ──────────────────
    final_values = state.values if state else full_state
    messages = final_values.get("messages", [])
    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]

    if ai_msgs:
        last_ai = ai_msgs[-1].content
        route = final_values.get("route", "chat")
        msg_type = "linkedin" if route == "linkedin" else "chat"
        yield _sse({"type": msg_type, "content": last_ai})
    else:
        yield _sse({"type": "chat", "content": "Sorry, I couldn't generate a response."})

    yield _sse({"type": "done"})


# =============================================================================
# Request models
# =============================================================================

class ChatRequest(BaseModel):
    message:   str
    thread_id: str = ""


class HitlRequest(BaseModel):
    answers:   dict
    thread_id: str


# =============================================================================
# Endpoints
# =============================================================================

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
    Resume from interrupt. No manual state reconstruction needed.
    The graph resumes from exactly where it paused (inside the clarifier node).
    Command(resume=value) tells LangGraph to resume the interrupted node.
    The interrupt() call inside clarifier_node returns req.answers.
    """
    resume_input = Command(resume=req.answers)
    return StreamingResponse(
        _run_stream(resume_input, req.thread_id),
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
    """Non-blocking health check using async HTTP client."""
    status = {"graph": _agent_graph is not None, "ollama": False}
    try:
        import httpx
        from config import OLLAMA_BASE_URL
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            status["ollama"] = r.status_code == 200
    except Exception:
        pass
    return status


# =============================================================================
# SPA serving
# =============================================================================

from pathlib import Path
from fastapi.staticfiles import StaticFiles

UI_DIR   = Path(__file__).parent / "ui" / "dist"
UI_INDEX = UI_DIR / "index.html"

# Mount static assets (js, css, images) with correct MIME types.
# This MUST come before the SPA catch-all so /assets/index-xxx.js
# is served as application/javascript, not text/html.
if (UI_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="static-assets")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if UI_INDEX.exists():
        return HTMLResponse(UI_INDEX.read_text())
    return HTMLResponse("<h1>UI not built. Run: cd ui && npm run build</h1>")


@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_spa(path: str):
    # Serve actual files from ui/dist if they exist (e.g. favicon, robots.txt)
    file_path = UI_DIR / path
    if file_path.is_file():
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    # Otherwise fall back to index.html for SPA client-side routing
    if UI_INDEX.exists():
        return HTMLResponse(UI_INDEX.read_text())
    return HTMLResponse("<h1>Not found</h1>", status_code=404)
