# Personal AI Assistant — Multi-Agent System

A locally-running multi-agent personal assistant built with LangGraph, Ollama, PostgreSQL, and ChromaDB.
No cloud APIs. No paid models. Everything runs on your machine.

---

## What It Does

- **Chat** — General assistant with automatic web search when needed. Knows who you are via long-term memory.
- **LinkedIn Post Generator** — Full pipeline: web research → HITL clarification → draft → critique → style match.
- **RAG Search** — Ask questions against your own documents (resume, notes, anything you ingest).
- **Long-Term Memory (LTM)** — Automatically extracts and remembers facts about you across sessions.
- **Short-Term Memory (STM)** — Rolling summarization keeps conversations coherent without blowing the context window.

More agents (email, tweets, calendar) plug in without changing the core architecture.

---

## Architecture Overview

```
User Message
    │
    ▼
[memory_inject]  ← loads STM summary + LTM facts from store
    │
    ▼
[router]  ← two-tier: regex fast-path → LLM structured output fallback
    │         outputs: route + research_mode + search_queries
    │
    ├─ linkedin ──► [linkedin subgraph]
    │                researcher  (web + LTM + ChromaDB, all concurrent)
    │                clarifier   (interrupt/resume for HITL)
    │                generator → evaluator → style_matcher
    │
    ├─ rag ───────► [rag subgraph]
    │                researcher  (web + LTM + ChromaDB, concurrent)
    │                search_response
    │
    └─ chat ──────► [chat subgraph]
                     chatbot     (internal web search trigger, no router overhead)
    │
    ▼
[memory_update]  ← LTM extraction + STM summarization check
    │
    ▼
   END
```

### Router Research Modes (linkedin + rag only)

| Mode | When | What happens |
|------|------|-------------|
| `closed_book` | Evergreen topic, no external facts needed | Skip web search entirely |
| `hybrid` | Topic needs current examples or event context | Web search runs concurrent with LTM |
| `open_book` | News, rankings, company info, live data | Full web search, strict recency filter |

The router outputs `route`, `research_mode`, and `search_queries` in a single structured LLM call.
Chat always goes directly to the chatbot — no research classification overhead.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent framework | LangGraph 0.2+ | Subgraphs, interrupt/resume HITL, store injection |
| LLM — writing | Ollama llama3.1:8b | Best local quality for creative generation |
| LLM — structured | Ollama qwen2.5:7b | Fast reliable JSON at temp=0 |
| LLM — routing | Ollama llama3.2:3b | Minimal latency for classification |
| Embeddings | Ollama nomic-embed-text | 768-dim, fast, local |
| STM checkpoints | PostgreSQL + AsyncPostgresSaver | LangGraph native, session history, time-travel |
| LTM store | LangGraph InMemoryStore → pgvector | store injection keeps graph clean |
| RAG documents | ChromaDB persistent | Cosine similarity over personal docs |
| Web search | DuckDuckGo (no key needed) | Free, local-friendly |
| Backend | FastAPI + SSE | Real async streaming |
| Frontend | React + Vite | Clean UI, proper HITL forms |

**NOT used:**
- Redis — LLM calls take 2-8s. A 1ms cache changes nothing. One less service to run locally.
- FAISS directly — ChromaDB uses it internally. Raw FAISS has no persistence or metadata.
- Cloud LLMs — 100% local, zero API cost.

---

## Project Structure

```
project/
├── agents/
│   ├── __init__.py
│   ├── router.py            # Two-tier router: regex → LLM RouterDecision
│   └── memory_nodes.py      # memory_inject + memory_update graph nodes
│
├── subgraphs/
│   ├── __init__.py
│   ├── chat.py              # Chat subgraph (1 node)
│   ├── linkedin.py          # LinkedIn subgraph (researcher→clarifier→gen→eval→style)
│   └── rag.py               # RAG subgraph (researcher→search_response)
│
├── memory/
│   ├── __init__.py
│   ├── stm.py               # STM summarization helpers
│   └── ltm.py               # LTM extract + retrieve via LangGraph store
│
├── tools/
│   ├── __init__.py
│   ├── web_search.py        # Async DuckDuckGo wrapper
│   └── rag_engine.py        # ChromaDB init, ingest, retrieve
│
├── data/
│   ├── documents/
│   │   ├── facts/           # Your resume, bio, skills (.txt / .pdf)
│   │   └── linkedin/        # Past LinkedIn posts (.txt) for style matching
│   └── chroma_db/           # Auto-created by ChromaDB on first ingest
│
├── ui/                      # React + Vite frontend
│
├── state.py                 # GlobalState (checkpoint) + LocalState TypedDicts
├── config.py                # All env vars and constants — single source of truth
├── llm_config.py            # Three LLM tier instances
├── graph.py                 # Main graph assembly + get_graph_async()
├── server.py                # FastAPI backend with SSE streaming
├── ingest.py                # One-time: embed your documents into ChromaDB
├── setup_db.py              # One-time: create PostgreSQL tables + pgvector
│
├── .env                     # Your secrets — never commit this
├── .env.example             # Copy → .env and fill in
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1 — Pull Ollama models

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Make sure Ollama is running: `ollama serve`

### 2 — Start PostgreSQL via Docker

```bash
docker run --name pa-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=postgres \
  -p 5442:5432 \
  -d postgres:16
```

To restart after a reboot: `docker start pa-postgres`

### 3 — Python environment

```bash
cd ~/project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4 — Environment variables

```bash
cp .env.example .env
# Open .env and update any values you changed (e.g. different DB password)
```

### 5 — Initialize the database

```bash
python setup_db.py
```

Creates the `ltm_facts` table and enables pgvector.

### 6 — Add your personal data (recommended)

Put your documents in:
- `data/documents/facts/` — resume.txt, bio.txt, skills.txt, any project descriptions
- `data/documents/linkedin/` — past_post_1.txt, past_post_2.txt (one post per file)

Then ingest:
```bash
python ingest.py
```

You can re-run `ingest.py` anytime after adding new documents — it upserts safely.

### 7 — Start the backend

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 8 — Start the frontend (separate terminal)

```bash
cd ui/
npm install
npm run dev
```

Open `http://localhost:5173`

---

## How to Use

**Chat:** Just type anything. Ask about news, concepts, your day. The assistant automatically searches the web when it detects the question needs current information.

**LinkedIn post:** Say things like:
- "Write a LinkedIn post about my 10th rank in IRC"
- "Draft a post about finishing my ML project"
- "Create a post announcing I joined XYZ company"

The system will look up the event/context on the web, then ask you specific questions it can't answer from research (your role, your contribution, specific outcomes). Then it writes, critiques, and style-matches to your past posts.

**RAG search:** Say things like:
- "What skills do I have in my profile?"
- "Search my resume for ML experience"
- "What projects have I done with Python?"

---

## Adding a New Agent (e.g. email writer)

1. Create `subgraphs/email.py` — build a `StateGraph` over `LocalState`, compile to `email_subgraph`
2. Add `"email"` to `RouteEnum` in `state.py`
3. Add regex fast-path in `agents/router.py` → `_fast_route()`
4. Wire it in `graph.py` conditional edges: `"email": email_subgraph`
5. Done — memory injection and LTM extraction are automatic

---

## Key Design Decisions

**Why subgraphs?**
Each agent pipeline is self-contained. The main graph only routes and handles memory. Adding agent #4 never touches agents #1–3. Clean separation of concerns.

**Why `interrupt()` for HITL instead of re-routing to START?**
The old approach discarded all research results and re-ran everything from scratch when the user answered. `interrupt()` genuinely pauses at the checkpoint. `Command(resume=answers)` continues from exactly that point — web research, LTM context, everything is preserved.

**Why does web research happen BEFORE the clarifier for LinkedIn?**
The clarifier asks better questions when it already knows what the event is. "What was your role on the team?" is a bad question when the assistant doesn't know what IRC is. After researching "IRC robotics competition 2025" and learning the event format, it can ask "Did you work on the navigation subsystem or mechanical?" — which is specific and useful.

**Why is chat's web search inside the chatbot, not the router?**
The router fires on every single message. Routing classification adds ~300ms. For chat, that overhead on every "hi" or "explain backprop" is unacceptable. The chatbot uses a fast regex signal detector internally — zero overhead for non-web messages, web search only when the pattern clearly matches.

**Why structured output everywhere?**
`model.with_structured_output(PydanticModel)` at `temperature=0` eliminates hallucinated JSON, unparseable responses, and brittle regex on LLM output. Every decision node produces a typed Pydantic model. This is the single biggest anti-hallucination measure in the system.

---

## Approximate Latency (llama3.1:8b, consumer hardware)

| Path | Time |
|------|------|
| Chat (no web) | 3–6s |
| Chat (with web search) | 5–9s |
| LinkedIn (closed_book) | 15–20s |
| LinkedIn (hybrid/open_book) | 18–28s |
| RAG search | 5–10s |

Web search and LTM retrieval always run concurrently — they do not stack.
