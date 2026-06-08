# Architecture Redesign — Personal AI Assistant
# Implementation Guide for Coding Agent

> **Context**: This project is being redesigned from scratch. The old architecture has a
> "God State" anti-pattern (single 20+ field TypedDict used everywhere), broken subgraph
> isolation, and a fragile state-flag HITL workaround. This plan fixes all of that.

---

## Executive Summary

Build a LangGraph architecture where each capability (Chat, RAG, LinkedIn) is a truly
isolated subgraph with its own private state. The parent orchestrator holds only 8 shared
fields. LinkedIn HITL uses LangGraph's native `interrupt()` instead of state-flag booleans.

---

## 1. Target Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                 PARENT GRAPH (OrchestratorState)                  │
│                                                                   │
│  START → memory_inject → router →─┬─► chat_subgraph     ─┬→ memory_update → END
│                                   ├─► rag_subgraph      ─┤
│                                   └─► linkedin_subgraph ─┘
│                                                                   │
│  OrchestratorState fields (8 total):                              │
│    user_id, messages, stm_summary, ltm_context,                   │
│    route, research_mode, search_queries                           │
└───────────────────────────────────────────────────────────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
   ┌───────────┐      ┌────────────┐      ┌──────────────────┐
   │ ChatState │      │  RAGState  │      │  LinkedInState   │
   │           │      │            │      │                  │
   │ messages  │      │ messages   │      │ messages         │ ← overlapping
   │ stm_sum   │      │ ltm_ctx   │      │ ltm_ctx          │   fields mapped
   │ ltm_ctx   │      │ r_mode    │      │ r_mode           │   automatically
   │           │      │ s_queries │      │ s_queries        │
   │           │      │───────────│      │──────────────────│
   │           │      │ chroma_res│      │ web_results      │ ← private fields
   │           │      │ web_res   │      │ style_examples   │   stay inside
   │           │      │           │      │ draft, revised   │   the subgraph
   │           │      │           │      │ final_post       │
   │           │      │           │      │ hitl_answers     │
   └───────────┘      └────────────┘      └──────────────────┘
```

### How LangGraph subgraph state mapping works

When you add a compiled subgraph as a node to the parent graph, LangGraph automatically:
1. **On entry**: copies parent state values into the subgraph for overlapping field names
2. **On exit**: copies subgraph values back into the parent for overlapping field names

Fields only in the subgraph schema are private — never checkpointed in the parent.

### HITL via interrupt()

The LinkedIn clarifier calls `interrupt(questions)`. The graph pauses, checkpoint is saved.
When the server calls `Command(resume=answers)`, the `interrupt()` call returns the answers
and the node continues. No re-running router, memory, or researcher.

---

## 2. Target Directory Structure

```
project/
├── config.py                    # KEEP — env vars, single source of truth
├── llm_config.py                # KEEP — three-tier LLM instances
├── shared.py                    # NEW  — shared utilities (keywords, helpers)
│
├── states/                      # NEW  — all state schemas
│   ├── __init__.py              #        re-exports all state classes
│   ├── orchestrator.py          #        OrchestratorState (8 fields)
│   ├── chat.py                  #        ChatState (3 fields)
│   ├── rag.py                   #        RAGState (6 fields)
│   └── linkedin.py              #        LinkedInState (11 fields)
│
├── nodes/                       # NEW (replaces agents/)
│   ├── __init__.py
│   ├── memory.py                #        memory_inject_node, memory_update_node
│   └── router.py                #        router_node, route_decision
│
├── subgraphs/                   # REWRITE — isolated state per subgraph
│   ├── __init__.py
│   ├── chat.py                  #        ChatState nodes + build_chat_subgraph()
│   ├── rag.py                   #        RAGState nodes  + build_rag_subgraph()
│   └── linkedin.py              #        LinkedInState nodes + build_linkedin_subgraph()
│
├── memory/                      # KEEP — no structural changes
│   ├── __init__.py
│   ├── stm.py
│   └── ltm.py
│
├── tools/                       # KEEP — no changes needed
│   ├── __init__.py
│   ├── web_search.py
│   └── rag_engine.py
│
├── graph.py                     # REWRITE — parent orchestrator
├── server.py                    # REWRITE — simplified HITL
├── ingest.py                    # KEEP
├── setup_db.py                  # KEEP
├── data/                        # KEEP
└── ui/                          # KEEP
```

---

## 3. Files to Delete After Migration

- `state.py` — replaced by `states/` directory
- `agents/` directory — replaced by `nodes/` directory
- `app.py` — unrelated GraphRAG hackathon demo (back up if desired)
- `=1.35.0` — stray file from botched pip install
- All `__pycache__/` directories

---

## 4. State Schema Definitions

### 4a. states/orchestrator.py

```python
"""
OrchestratorState — the ONLY state the parent graph checkpoints.

Rule: if a field is not needed by memory, routing, or cross-subgraph
communication, it does NOT belong here.
"""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class OrchestratorState(TypedDict):
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    stm_summary: str
    ltm_context: str
    route: str                # "chat" | "rag" | "linkedin"
    research_mode: str        # "closed_book" | "hybrid" | "open_book"
    search_queries: list[str]
```

### 4b. states/chat.py

```python
"""ChatState — overlapping fields only. No private fields needed."""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    stm_summary: str
    ltm_context: str
```

### 4c. states/rag.py

```python
"""RAGState — overlapping + 2 private fields for retrieval results."""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ltm_context: str
    research_mode: str
    search_queries: list[str]
    # Private
    chroma_results: str
    web_results: str
```

### 4d. states/linkedin.py

```python
"""LinkedInState — overlapping + private fields for the generation pipeline."""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class LinkedInState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ltm_context: str
    research_mode: str
    search_queries: list[str]
    # Private: Research
    web_results: str
    style_examples: str
    # Private: Generation
    fact_list: str
    draft: str
    revised_draft: str
    final_post: str
    # Private: HITL
    hitl_answers: dict
```

### 4e. states/__init__.py

```python
from states.orchestrator import OrchestratorState
from states.chat import ChatState
from states.rag import RAGState
from states.linkedin import LinkedInState
```

---

## 5. Shared Utilities

### shared.py

```python
"""
shared.py — Utility functions used across multiple subgraphs and nodes.
Single source of truth. Import from here, never copy-paste.
"""
import re
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


KW_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "is","are","was","were","be","been","i","my","me","we","you","your","it","its",
    "what","can","could","would","should","have","has","had","do","did",
    "please","help","need","want","like","tell","show","give","find",
    "hi","hey","hello","ok","okay","yes","no","got","sure","this","that","which",
    "write","draft","create","generate","make","compose","craft","post",
    "linkedin","about","search","look","check","list","get","fetch","retrieve",
}

LTM_SKIP_PATTERNS = [
    r"^(hi|hey|hello|good\s+morning|good\s+evening|good\s+night|sup|yo)[\s!.,]*$",
    r"^(ok|okay|thanks|thank\s+you|sure|yes|no|got\s+it|great|cool|nice|sounds\s+good)[\s!.,]*$",
    r"^(what|how|why|when|where|who|is|are|can|could|would|should|do|did)\b[^.!]*\?\s*$",
    r"\b(what\s+do\s+you\s+think|your\s+opinion|do\s+you\s+like|how\s+about\s+you)\b",
]


def extract_keywords(text: str, max_kw: int = 6) -> list[str]:
    """Extract content keywords, filtering stopwords."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    seen, out = set(), []
    for w in cleaned.split():
        if w not in KW_STOP and len(w) > 2 and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) >= max_kw:
                break
    return out


def build_rag_queries(text: str) -> list[str]:
    """Build 1-2 keyword-based search queries from text."""
    kw = extract_keywords(text)
    if not kw:
        return [text[:80]]
    queries = [" ".join(kw[:4])]
    if len(kw) > 4:
        queries.append(" ".join(kw[4:]))
    return queries


def last_human_content(messages: list[BaseMessage]) -> str:
    """Return the content of the last HumanMessage. Never returns AI content."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def last_ai_content(messages: list[BaseMessage]) -> str:
    """Return the content of the last AIMessage."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
    return ""


def is_trivial_message(text: str) -> bool:
    """Check if a message is too trivial for LTM extraction."""
    if len(text.split()) < 3:
        return True
    return any(re.search(p, text.strip(), re.IGNORECASE) for p in LTM_SKIP_PATTERNS)
```

---

## 6. Phase-by-Phase Build Instructions

Execute each phase fully before moving to the next. Verify at each step.

---

### PHASE 1 — Project Scaffold & Cleanup

**Goal**: Create the new directory structure and state schemas.

**Steps**:
1. Create directory `states/` with files:
   - `states/__init__.py` (Section 4e above)
   - `states/orchestrator.py` (Section 4a above)
   - `states/chat.py` (Section 4b above)
   - `states/rag.py` (Section 4c above)
   - `states/linkedin.py` (Section 4d above)

2. Create directory `nodes/` with empty `nodes/__init__.py`

3. Create `shared.py` (Section 5 above)

4. Delete `=1.35.0` from project root

**DO NOT delete old files yet.**

**Verify**:
```bash
python -c "from states import OrchestratorState, ChatState, RAGState, LinkedInState; print('OK')"
python -c "from shared import extract_keywords, last_human_content; print('OK')"
```

---

### PHASE 2 — Chat Subgraph (Isolated)

**Goal**: Rewrite `subgraphs/chat.py` using `ChatState` and a factory function.

**Graph**: `START → chatbot → END`

**What to port from old `subgraphs/chat.py`**:
- KEEP all regex patterns: `_WEB_SKIP`, `_WEB_TRIGGERS`, `_AFFIRMATIVE`, `_SEARCH_OFFER`
- KEEP all functions: `_needs_web()`, `_extract_search_topic()`, `_build_search_query()`, `_result_is_relevant()`
- KEEP system prompts: `_BASE_SYSTEM`, `_WEB_GROUNDING_SYSTEM`
- KEEP core chatbot_node logic (web search, message trimming, LLM call, grounding)

**What to change**:
- `GlobalState` → `ChatState` in type annotations
- `from state import GlobalState` → `from states.chat import ChatState`
- Replace local `_last_human_content` with `from shared import last_human_content`
- Remove module-level `_g = StateGraph(...)` and `chat_subgraph = _g.compile()`
- Add factory function:

```python
from langgraph.graph import StateGraph, START, END
from states.chat import ChatState

def build_chat_subgraph():
    g = StateGraph(ChatState)
    g.add_node("chatbot", chatbot_node)
    g.add_edge(START, "chatbot")
    g.add_edge("chatbot", END)
    return g.compile()
```

**Node signature**: `async def chatbot_node(state: ChatState) -> dict:`
**Output**: `{"messages": [AIMessage(content=reply)]}`

---

### PHASE 3 — RAG Subgraph (Isolated)

**Goal**: Rewrite `subgraphs/rag.py` using `RAGState` and a factory function.

**Graph**: `START → researcher → synthesizer → END`

**What to port from old `subgraphs/rag.py`**:
- KEEP the concurrent retrieval pattern (ChromaDB + web via asyncio.gather)
- KEEP the `_SEARCH_RESPONSE_SYSTEM` prompt
- KEEP the source synthesis logic

**What to change**:
- `GlobalState` → `RAGState`
- `_rag_chroma_facts` → `chroma_results`
- `_rag_web_results` → `web_results`
- Remove duplicated `_keywords()`, `_KW_STOP`, `_rag_queries()` → use `from shared import extract_keywords, build_rag_queries`
- Remove redundant `_HM` re-imports inside function bodies
- `search_response_node` → rename to `synthesizer_node`
- Add factory function:

```python
from langgraph.graph import StateGraph, START, END
from states.rag import RAGState

def build_rag_subgraph():
    g = StateGraph(RAGState)
    g.add_node("researcher", researcher_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()
```

**Node signatures**:
```python
async def researcher_node(state: RAGState, config: RunnableConfig, *, store: BaseStore) -> dict:
    # Outputs: {"chroma_results": str, "web_results": str}

async def synthesizer_node(state: RAGState) -> dict:
    # Outputs: {"messages": [AIMessage(content=reply)]}
```

---

### PHASE 4 — LinkedIn Subgraph (Isolated + interrupt HITL)

**Goal**: Rewrite `subgraphs/linkedin.py` using `LinkedInState`, factory function, and
native `interrupt()` for HITL.

**Graph**: `START → researcher → clarifier → generator → evaluator → style_matcher → END`

**CRITICAL: No conditional edge to END.** The clarifier uses `interrupt()` to pause the
graph. When resumed, execution continues from inside the clarifier — generator, evaluator,
style_matcher then run normally.

**What to port**:
- KEEP system prompts for clarifier, generator, evaluator, style_matcher
- KEEP `EvalResult`, `EvalIssue` Pydantic schemas
- KEEP `_build_fact_list()` logic (adapt field names)
- KEEP `_is_non_answer()` helper

**What to change**:
- `GlobalState` → `LinkedInState`
- ALL `_li_*` field names → clean names: `web_results`, `style_examples`, `draft`, `revised_draft`, `final_post`, `hitl_answers`
- **DELETE** all HITL state-flag fields: `_li_needs_hitl`, `_li_hitl_complete`, `_li_hitl_rounds`, `_li_hitl_questions`
- **DELETE** conditional edge `_route_after_clarifier`
- **IMPLEMENT** `interrupt()` in clarifier:

```python
from langgraph.types import interrupt
from shared import last_human_content

async def clarifier_node(state: LinkedInState) -> dict:
    messages    = state.get("messages", [])
    user_input  = last_human_content(messages)
    ltm_context = state.get("ltm_context", "")
    web_results = state.get("web_results", "")

    all_answers = {}

    for round_num in range(2):  # max 2 HITL rounds
        needed, questions = await _evaluate_context(
            user_input, ltm_context, web_results, all_answers
        )
        if not needed or not questions:
            break
        # Each interrupt() call pauses the graph.
        # When resumed with Command(resume=answers_dict),
        # interrupt() returns the answers_dict.
        round_answers = interrupt({"questions": questions, "round": round_num + 1})
        all_answers.update(round_answers)

    return {"hitl_answers": all_answers}
```

**Factory function**:
```python
from langgraph.graph import StateGraph, START, END
from states.linkedin import LinkedInState

def build_linkedin_subgraph():
    g = StateGraph(LinkedInState)
    g.add_node("researcher",    researcher_node)
    g.add_node("clarifier",     clarifier_node)
    g.add_node("generator",     generator_node)
    g.add_node("evaluator",     evaluator_node)
    g.add_node("style_matcher", style_matcher_node)

    g.add_edge(START,            "researcher")
    g.add_edge("researcher",     "clarifier")
    g.add_edge("clarifier",      "generator")   # straight edge, not conditional
    g.add_edge("generator",      "evaluator")
    g.add_edge("evaluator",      "style_matcher")
    g.add_edge("style_matcher",  END)

    return g.compile()
```

**Node signatures**:
```python
async def researcher_node(state: LinkedInState, config: RunnableConfig, *, store: BaseStore) -> dict:
    # Outputs: {"web_results": str, "style_examples": str}

async def clarifier_node(state: LinkedInState) -> dict:
    # Uses interrupt() for HITL. Outputs: {"hitl_answers": dict}

async def generator_node(state: LinkedInState) -> dict:
    # Outputs: {"draft": str, "fact_list": str}

async def evaluator_node(state: LinkedInState) -> dict:
    # Outputs: {"revised_draft": str}

async def style_matcher_node(state: LinkedInState) -> dict:
    # Outputs: {"messages": [AIMessage(final_post)], "final_post": str}
```

---

### PHASE 5 — Router & Memory Nodes (Parent-Level)

**Goal**: Port router and memory nodes to `OrchestratorState`, removing all HITL bypass logic.

#### 5a. nodes/router.py

**Port from old `agents/router.py`**:
- KEEP: Two-tier routing (regex fast-path + LLM structured output)
- KEEP: `RouterDecision` Pydantic model, `_ROUTER_SYSTEM` prompt
- KEEP: `_LINKEDIN_PATTERNS`, `_RAG_PATTERNS`, `_fast_route()`
- KEEP: `_llm_classify()` function
- **DELETE**: The entire HITL resume bypass block (`if state.get("_li_hitl_complete")`)
- **CHANGE**: `GlobalState` → `OrchestratorState`
- **CHANGE**: `from state import GlobalState` → `from states.orchestrator import OrchestratorState`
- **CHANGE**: import shared utilities from `shared.py`

**Signatures**:
```python
async def router_node(state: OrchestratorState, config: RunnableConfig) -> dict:
    # Outputs: {"route": str, "research_mode": str, "search_queries": list}

def route_decision(state: OrchestratorState) -> str:
    return state.get("route", "chat")
```

#### 5b. nodes/memory.py

**Port from old `agents/memory_nodes.py`**:
- KEEP: LTM retrieval logic in `memory_inject_node`
- KEEP: STM summarization + RemoveMessage trimming in `memory_update_node`
- KEEP: LTM extraction calling `memory/ltm.py`
- **DELETE**: All HITL bypass logic (`_li_hitl_complete`, `is_hitl_first_run`, `_li_hitl_answers`)
- **CHANGE**: `GlobalState` → `OrchestratorState`
- **CHANGE**: Use `from shared import last_human_content, last_ai_content, extract_keywords, is_trivial_message`

**Signatures**:
```python
async def memory_inject_node(
    state: OrchestratorState, config: RunnableConfig, *, store: BaseStore
) -> dict:
    # Outputs: {"ltm_context": str}

async def memory_update_node(
    state: OrchestratorState, config: RunnableConfig, *, store: BaseStore
) -> dict:
    # Outputs: {"stm_summary": str, "messages": [RemoveMessage(...)]} (conditional)
```

---

### PHASE 6 — Parent Graph Assembly

**Goal**: Wire the parent orchestrator graph with subgraphs as nodes.

**File**: `graph.py` (overwrite old file)

```python
"""
graph.py — Parent orchestrator graph assembly.

START → memory_inject → router → [chat | rag | linkedin] → memory_update → END
"""
import logging
from langgraph.graph import StateGraph, START, END
from states.orchestrator import OrchestratorState
from nodes.router import router_node, route_decision
from nodes.memory import memory_inject_node, memory_update_node

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None, store=None):
    from subgraphs.chat import build_chat_subgraph
    from subgraphs.rag import build_rag_subgraph
    from subgraphs.linkedin import build_linkedin_subgraph

    chat_sg     = build_chat_subgraph()
    rag_sg      = build_rag_subgraph()
    linkedin_sg = build_linkedin_subgraph()

    g = StateGraph(OrchestratorState)

    g.add_node("memory_inject",  memory_inject_node)
    g.add_node("router",         router_node)
    g.add_node("chat",           chat_sg)
    g.add_node("rag",            rag_sg)
    g.add_node("linkedin",       linkedin_sg)
    g.add_node("memory_update",  memory_update_node)

    g.add_edge(START, "memory_inject")
    g.add_edge("memory_inject", "router")
    g.add_conditional_edges(
        "router", route_decision,
        {"chat": "chat", "rag": "rag", "linkedin": "linkedin"},
    )
    g.add_edge("chat",     "memory_update")
    g.add_edge("rag",      "memory_update")
    g.add_edge("linkedin", "memory_update")
    g.add_edge("memory_update", END)

    return g.compile(checkpointer=checkpointer, store=store)


async def get_graph_async():
    """Production graph with PostgreSQL checkpointer + InMemoryStore for LTM."""
    checkpointer = None
    store = None

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

    if store:
        try:
            from memory.ltm import load_ltm_from_postgres
            await load_ltm_from_postgres("default", store)
        except Exception as e:
            logger.warning(f"LTM load from PostgreSQL failed ({e}) — starting empty")

    return graph
```

**Verify**:
```bash
python -c "
from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph
g = build_graph(checkpointer=MemorySaver())
print('Nodes:', list(g.get_graph().nodes))
print('OK')
"
```

---

### PHASE 7 — Server Rewrite

**Goal**: Simplified server with interrupt-based HITL.

**File**: `server.py` (overwrite old file)

**Key changes**:

1. **Minimal initial state**:
```python
def _build_initial_state(message: str) -> dict:
    from langchain_core.messages import HumanMessage
    return {
        "user_id":  DEFAULT_USER,
        "messages": [HumanMessage(content=message)],
    }
```

2. **Streaming with interrupt detection** — after streaming completes, check
   `state.next` to detect if an interrupt occurred. If so, extract the interrupt
   value and send an `hitl` SSE event.

3. **HITL resume — radically simplified**:
```python
from langgraph.types import Command

@app.post("/api/hitl")
async def hitl(req: HitlRequest):
    resume_input = Command(resume=req.answers)
    return StreamingResponse(
        _run_stream(resume_input, req.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

4. **Fix async health check** — use `httpx.AsyncClient` instead of sync `httpx.get()`

5. **Keep these endpoints as-is** (no state dependency):
   - `GET /api/sessions`
   - `GET /api/session-messages` (update to work with `OrchestratorState`)
   - `GET /api/ltm`, `DELETE /api/ltm/{key}`
   - `GET /api/stats`
   - `GET /`, `GET /{path:path}` (SPA serving)

---

### PHASE 8 — Cleanup & Final Verification

1. **Delete old files**:
   - `state.py`
   - `agents/` directory
   - `app.py` (unrelated demo)
   - `=1.35.0`
   - All `__pycache__/`

2. **Update `requirements.txt`**: change `langgraph>=0.2.0` to `langgraph>=0.2.40`

3. **Verify all imports**:
```bash
python -c "from states import OrchestratorState, ChatState, RAGState, LinkedInState"
python -c "from nodes.router import router_node"
python -c "from nodes.memory import memory_inject_node"
python -c "from subgraphs.chat import build_chat_subgraph"
python -c "from subgraphs.rag import build_rag_subgraph"
python -c "from subgraphs.linkedin import build_linkedin_subgraph"
python -c "from graph import build_graph; from langgraph.checkpoint.memory import MemorySaver; g = build_graph(checkpointer=MemorySaver()); print('OK')"
```

4. **End-to-end tests**:
   - Chat: send "hello" → verify response
   - RAG: send "what are my skills?" → verify document retrieval
   - LinkedIn: send "write a LinkedIn post about my hackathon" → verify HITL interrupt → submit answers → verify post
   - Memory: verify LTM facts extracted after turns
   - STM: verify summary works after 10+ messages

---

## 7. Before vs. After

| Aspect | Before | After |
|---|---|---|
| Parent state fields | 20+ | 8 |
| Checkpointed per chat turn | 20+ (incl. empty `_li_*`) | 8 |
| HITL resume | Re-run 6 nodes | Resume mid-node |
| HITL state flags | 4 booleans | 0 |
| Initial state dict | 20+ fields | 3 fields |
| HITL endpoint | 60 lines | 10 lines |
| Duplicated utilities | 4 copies | 1 (shared.py) |
| Subgraph isolation | None | Complete |
| Import side effects | 3 compilations | 0 |

---

## 8. LangGraph Version Requirement

`interrupt()` requires LangGraph >= 0.2.39. `Command` requires >= 0.2.40.

```bash
pip show langgraph | grep Version
pip install --upgrade langgraph langgraph-checkpoint-postgres
```
