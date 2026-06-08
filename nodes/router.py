"""
nodes/router.py — Two-tier router for the parent orchestrator.

TIER 1 — Regex fast-path (< 1ms, no LLM)
  Detects linkedin and rag patterns. Returns "chat" as default.

TIER 2 — LLM structured output (llama3.2:3b, ~300ms)
  Only fires for linkedin/rag to get research_mode + queries.
  Chat bypasses entirely.

No HITL bypass needed — with interrupt(), the router never re-runs on resume.
"""

import re
import logging
from datetime import date
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from states.orchestrator import OrchestratorState
from shared import last_human_content
from config import ROUTER_MIN_CONFIDENCE
from llm_config import llm_router

logger = logging.getLogger(__name__)


# =============================================================================
# Structured output schema
# =============================================================================

class RouterDecision(BaseModel):
    research_mode: str = Field(
        default="closed_book",
        description=(
            "MUST be one of: closed_book, hybrid, open_book. "
            "This is NOT the route. It classifies web search need."
        )
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 short factual search queries about the SUBJECT. "
            "DO NOT include dates or years as prefixes in the queries. "
            "Just the search terms. Empty for closed_book."
        )
    )
    confidence: float = Field(description="Confidence 0.0-1.0.")
    reason: str = Field(description="One sentence explaining the research_mode decision.")


# =============================================================================
# Tier 1 — Regex fast-path
# =============================================================================

_LINKEDIN_PATTERNS = [
    r"\blinkedin\b",
    r"\b(write|draft|create|generate|make|compose|craft|prepare)\b.{0,35}\bpost\b",
    r"\bpost\b.{0,20}\b(about|for|on|announcing|regarding)\b",
    r"\b(announce|announcing)\b.{0,30}\b(i\s+(joined|got|started|built|won|finished|completed|placed))\b",
]

_RAG_PATTERNS = [
    r"\bmy\s+(skills?|experience|projects?|achievements?|background|resume|profile|education|portfolio|cv)\b",
    r"\b(search|look\s*up|find)\s+(my|in\s+my)\b",
    r"\b(in|from)\s+my\s+(database|db|docs|files|profile|resume|notes|documents)\b",
    r"\bwhat\s+(do\s+i\s+know|have\s+i\s+done|skills\s+do\s+i\s+have|experience\s+do\s+i\s+have)\b",
    r"\bcheck\s+my\s+(profile|resume|database|skills|background)\b",
    r"\bshow\s+me\s+my\b",
    r"\blist\s+my\b",
]


def _fast_route(text: str) -> str:
    t = text.lower().strip()
    if any(re.search(p, t, re.IGNORECASE) for p in _LINKEDIN_PATTERNS):
        return "linkedin"
    if any(re.search(p, t, re.IGNORECASE) for p in _RAG_PATTERNS):
        return "rag"
    return "chat"


# =============================================================================
# Tier 2 — LLM classification (research_mode + queries only)
# =============================================================================

_ROUTER_SYSTEM = """\
You are classifying whether a web search would help write a LinkedIn post or answer a query.

YOUR ONLY JOB: Set research_mode to closed_book, hybrid, or open_book.
Do NOT decide the route — that is already determined.

DEFINITIONS:
  closed_book — Web search would return NOTHING useful.
    Use when the subject is personal/internal:
    - "my college hackathon", "my university project", "my college ML competition"
    - ANY event at a college/university (not publicly indexed)
    - Personal roles, personal results, personal projects
    - Anything where facts come from the USER, not the internet
    The clarifier will ask the user for these personal facts instead.

  hybrid — Web search returns SOME useful background context.
    Use when the subject is a real named public event/company that IS indexed:
    - Named national/international competitions (IRC, IEEE, ICPC, Google HashCode)
    - Named companies the user mentions joining
    - Named public technologies or products

  open_book — Web search is ESSENTIAL.
    Use for current events, news, market data, public rankings, live scores.

CRITICAL RULE: College/university events are ALWAYS closed_book.
The web does not have results for your specific college hackathon or your
university ML competition. Don't waste time searching for them.

SEARCH QUERIES (hybrid/open_book only):
  Write 2-4 short search queries about the subject — just keywords/phrases.
  Example good query: "IRC robotics competition format ranking"
  Example BAD query: "March 14 2026: What is the IRC competition?"
  Never prefix queries with dates or "what is" questions — just terms.

OUTPUT: Strict JSON. No extra text.\
"""

_llm_router_structured = llm_router.with_structured_output(RouterDecision)


async def _llm_classify(user_input: str, context: str) -> RouterDecision:
    year = date.today().year
    try:
        decision: RouterDecision = await _llm_router_structured.ainvoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=(
                f"Current year: {year}\n\n"
                f"Context:\n{context}\n\n"
                f"Message to classify: {user_input}"
            )),
        ])
        return decision
    except Exception as e:
        logger.warning(f"[router] LLM classify failed ({e}) — defaulting closed_book")
        return RouterDecision(
            research_mode="closed_book",
            search_queries=[],
            confidence=0.5,
            reason=f"LLM failed: {e}",
        )


# =============================================================================
# Graph node
# =============================================================================

async def router_node(state: OrchestratorState, config: RunnableConfig) -> dict:
    messages   = state.get("messages", [])
    user_input = last_human_content(messages)

    if not user_input.strip():
        return {"route": "chat", "research_mode": "closed_book", "search_queries": []}

    route = _fast_route(user_input)

    # Chat: no research classification needed
    if route == "chat":
        logger.debug("[router] fast → chat")
        return {
            "route":          "chat",
            "research_mode":  "closed_book",
            "search_queries": [],
        }

    # linkedin/rag: call LLM for research_mode + queries
    context_parts = []
    stm = state.get("stm_summary", "")
    if stm:
        context_parts.append(f"Earlier: {stm[:150]}")
    recent = messages[-3:] if len(messages) > 3 else messages[:-1]
    for msg in recent:
        if isinstance(msg, HumanMessage):
            context_parts.append(f"User: {msg.content[:80]}")
        elif isinstance(msg, AIMessage):
            context_parts.append(f"AI: {msg.content[:60]}")
    context = "\n".join(context_parts) or "(first message)"

    decision = await _llm_classify(user_input, context)

    search_queries = (
        decision.search_queries
        if decision.research_mode in ("hybrid", "open_book")
        else []
    )

    logger.info(
        f"[router] → {route} | {decision.research_mode} | "
        f"{len(search_queries)} queries | confidence={decision.confidence:.2f}"
    )

    return {
        "route":          route,
        "research_mode":  decision.research_mode,
        "search_queries": search_queries,
    }


def route_decision(state: OrchestratorState) -> str:
    return state.get("route", "chat")
