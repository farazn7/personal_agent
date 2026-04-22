"""
subgraphs/linkedin.py — LinkedIn post generation subgraph.

Pipeline:
  researcher -> clarifier -> (conditional) -> generator -> evaluator -> style_matcher
                                |
                                +-- if _li_needs_hitl=True --> END
                                    server.py emits hitl SSE
                                    user answers → /api/hitl with merged answers

MULTI-TURN CLARIFIER:
  Up to 2 rounds of HITL. On each resume, clarifier re-evaluates with all
  accumulated answers. If still insufficient and rounds < 2, asks again.
  After 2 rounds, proceeds regardless (user can always provide more in the post).

HITL DESIGN — STATE FLAGS (no interrupt()):
  interrupt() is broken in this LangGraph version. State booleans instead.
"""

import re
import json
import asyncio
import logging
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore

from state import GlobalState
from llm_config import llm_write, llm_precise
from tools.web_search import web_search_async
from tools.rag_engine import retrieve_linkedin_examples

logger = logging.getLogger(__name__)


# =============================================================================
# Shared utilities
# =============================================================================

_KW_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "is","are","was","were","be","been","i","my","me","we","you","your","it","its",
    "what","can","could","would","should","have","has","had","do","did","please",
    "help","need","want","like","tell","show","give","find","write","draft",
    "create","generate","make","compose","craft","post","linkedin","about",
}


def _last_human_content(messages: list) -> str:
    """Return last HumanMessage content. Never accidentally returns an AIMessage."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def _keywords(text: str, max_kw: int = 6) -> list:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    seen, out = set(), []
    for w in cleaned.split():
        if w not in _KW_STOP and len(w) > 2 and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) >= max_kw:
                break
    return out


def _rag_queries(text: str) -> list:
    kw = _keywords(text)
    if not kw:
        return [text[:80]]
    queries = [" ".join(kw[:4])]
    if len(kw) > 4:
        queries.append(" ".join(kw[4:]))
    return queries


# =============================================================================
# 1. Researcher node
# =============================================================================

async def researcher_node(
    state: GlobalState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Concurrent: web search (if hybrid/open_book) + ChromaDB style examples.
    Skips on HITL resume — all research data is already in the checkpoint.
    """
    if state.get("_li_hitl_complete", False):
        logger.debug("[researcher] HITL resume — data already in checkpoint, skipping")
        return {"current_agent": "Researcher"}

    messages       = state.get("messages", [])
    user_input     = _last_human_content(messages)
    research_mode  = state.get("research_mode", "closed_book")
    search_queries = state.get("search_queries", [])
    queries        = search_queries or _rag_queries(user_input)
    loop           = asyncio.get_running_loop()

    async def _style_examples() -> str:
        return await loop.run_in_executor(None, retrieve_linkedin_examples, queries, 3)

    async def _web() -> str:
        if research_mode in ("hybrid", "open_book") and queries:
            return await web_search_async(queries[:3], max_results=4)
        return ""

    style_examples, web_results = await asyncio.gather(
        _style_examples(), _web(), return_exceptions=True
    )
    if isinstance(style_examples, Exception):
        logger.warning(f"[researcher] ChromaDB failed: {style_examples}")
        style_examples = ""
    if isinstance(web_results, Exception):
        logger.warning(f"[researcher] web search failed: {web_results}")
        web_results = ""

    logger.info(
        f"[linkedin researcher] mode={research_mode} "
        f"web={'yes' if web_results else 'no'} "
        f"examples={'yes' if style_examples else 'no'}"
    )
    return {
        "_li_web_results":    web_results,
        "_li_style_examples": style_examples,
        "current_agent":      "Researcher",
    }


# =============================================================================
# 2. Clarifier — multi-turn state-flag HITL
# =============================================================================

_CLARIFIER_SYSTEM = """\
You are a LinkedIn post assistant. Check if you have enough SPECIFIC, CONFIRMED
facts to write a grounded post without inventing anything.

You receive:
  - The user's original request
  - Web research (may be empty)
  - Memory facts
  - Accumulated answers from previous questions (may be empty on first run)

STEP 1 — Inventory what you actually have:
  WHAT   — specific description of what happened/was built/was done
  ROLE   — what the user personally did (their contribution)
  RESULT — concrete outcome (rank, metric, deployment, award)
  TECH   — tools/stack (only if technical is the point)

STEP 2 — Decide:
  - Count how many of the above you have confirmed facts for.
  - If 2+ are STILL missing (not answerable from any source): needed=true.
  - If you can write something honest and specific: needed=false.
  - For opinion/thought-leadership posts: always needed=false.
  - If you already asked about something and the user answered, do NOT ask again.

RULES:
  - Do NOT ask about info already in the user's message, memory, or prior answers.
  - Questions must be specific with a concrete example answer in ().
  - MAX 2 questions per round.

OUTPUT: Valid JSON only.
{"needed": true/false, "questions": ["question (e.g. example)"]}\
"""


async def clarifier_node(state: GlobalState) -> dict:
    """
    Multi-turn state-flag HITL clarifier. No interrupt() used.

    First run:
      LLM evaluates available context. If questions needed, sets
      _li_needs_hitl=True → conditional edge routes to END.
      server.py emits hitl SSE with the questions.

    Resume run (_li_hitl_complete=True, _li_hitl_rounds=N):
      Re-evaluates with all accumulated answers so far.
      If still needs more AND rounds < 2: asks again (new questions).
      If sufficient OR rounds >= 2: clears flag → routes to generator.
    """
    messages     = state.get("messages", [])
    user_input   = _last_human_content(messages)
    ltm_context  = state.get("ltm_context", "")
    web_results  = state.get("_li_web_results", "")
    hitl_answers = state.get("_li_hitl_answers", {}) or {}
    hitl_rounds  = state.get("_li_hitl_rounds", 0)
    is_resume    = state.get("_li_hitl_complete", False)

    # Hard limit: max 2 rounds of HITL regardless
    if is_resume and hitl_rounds >= 2:
        logger.info("[clarifier] max HITL rounds reached — proceeding to generator")
        return {
            "_li_needs_hitl":    False,
            "_li_hitl_complete": False,
            "current_agent":     "Clarifier",
        }

    known_facts = ltm_context.strip() or "(none)"
    web_context = web_results.strip() or "(no web research)"

    # Build context including accumulated answers (for re-evaluation on resume)
    answers_section = ""
    if hitl_answers:
        lines = ["User's answers to previous questions:"]
        for q, a in hitl_answers.items():
            answer_text = str(a).strip()
            # Only include genuine answers, not confused non-answers
            if answer_text and not _is_non_answer(answer_text):
                lines.append(f"  Q: {q}")
                lines.append(f"  A: {answer_text}")
        if len(lines) > 1:
            answers_section = "\n".join(lines)

    content = (
        f"User's original request: {user_input}\n\n"
        f"Web research:\n{web_context[:600]}\n\n"
        f"Known facts from memory:\n{known_facts}\n\n"
    )
    if answers_section:
        content += f"{answers_section}\n\n"
    content += "Output JSON:"

    try:
        resp = await llm_precise.ainvoke([
            SystemMessage(content=_CLARIFIER_SYSTEM),
            HumanMessage(content=content),
        ])
        raw   = resp.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON")
        result    = json.loads(raw[start:end])
        needed    = bool(result.get("needed", False))
        questions = [q.strip() for q in result.get("questions", []) if q.strip()][:2]
        if needed and not questions:
            needed = False
    except Exception as e:
        logger.warning(f"[clarifier] LLM failed ({e}) — proceeding")
        needed, questions = False, []

    if needed and questions:
        logger.info(f"[clarifier] HITL round {hitl_rounds + 1}: {len(questions)} questions")
        return {
            "_li_needs_hitl":     True,
            "_li_hitl_questions": questions,
            "_li_hitl_complete":  False,   # reset so next resume re-enters clarifier
            "current_agent":      "Clarifier",
        }

    logger.debug(f"[clarifier] sufficient context (round {hitl_rounds}) — proceeding")
    return {
        "_li_needs_hitl":     False,
        "_li_hitl_questions": [],
        "_li_hitl_complete":  False,
        "current_agent":      "Clarifier",
    }


def _route_after_clarifier(state: GlobalState) -> str:
    if state.get("_li_needs_hitl", False):
        return END
    return "generator"


def _is_non_answer(text: str) -> bool:
    """Detect confused non-answers that would cause hallucination if treated as facts."""
    return (
        text.endswith("?") or
        bool(re.search(
            r"^(umm?|uh|hmm|idk|i don.t know|what do you|u want|you want|do you|which)",
            text, re.IGNORECASE
        ))
    )


# =============================================================================
# Fact list builder
# =============================================================================

def _build_fact_list(state: GlobalState) -> str:
    """
    Build the numbered fact list for the generator.
    Only includes confirmed, genuine information — no hallucination fodder.
    """
    items = []
    n     = 1

    messages   = state.get("messages", [])
    user_input = _last_human_content(messages)  # always the original request

    if user_input.strip():
        items.append(f"{n}. [USER SAID]: {user_input.strip()}")
        n += 1

    hitl_answers = state.get("_li_hitl_answers", {}) or {}
    for q, a in hitl_answers.items():
        answer_text = str(a).strip()
        if not answer_text or _is_non_answer(answer_text):
            logger.debug(f"[fact_list] skipping non-answer: '{answer_text[:40]}'")
            continue
        items.append(f"{n}. [USER CONFIRMED — {q[:60]}]: {answer_text}")
        n += 1

    web = state.get("_li_web_results", "")
    if web and web.strip():
        items.append(f"{n}. [WEB RESEARCH]: {web.strip()[:400]}")
        n += 1

    ltm = state.get("ltm_context", "")
    if ltm and ltm.strip():
        for line in ltm.splitlines():
            line = line.strip()
            if line.startswith(("•", "-", "*", "–")):
                fact = line.lstrip("•-*– ").strip()
                if fact and len(fact) > 10:
                    items.append(f"{n}. [MEMORY]: {fact}")
                    n += 1

    if not items:
        return (
            "NO FACTS AVAILABLE. Write ONLY what the user said. "
            "Do not add any details not explicitly stated.\n"
        )
    return "\n".join(items)


# =============================================================================
# 3. Generator node
# =============================================================================

_GENERATOR_SYSTEM = """\
You are a LinkedIn post writer. Write ONE post using ONLY the numbered facts.

HARD RULES — violation = automatic failure:
- Every sentence must trace to a numbered fact.
- DO NOT add: "prestigious", "renowned", "world-class", "highly competitive",
  "international", "global" unless the user's own words contain that word.
- DO NOT invent: team sizes, country counts, prize amounts, participants, dates,
  project names, technology names, or ANY detail not in the facts.
- Rank "10th" → write "10th". Not "top 10", not "top 10%".
- Use first person (I, My, We).
- NEVER start with: excited/thrilled/proud/honoured/humbled to share/announce
- IF A DETAIL IS MISSING: write around it using only what IS in the facts.
  "I built an ML solution" is fine. "I built 'Predictive Maintenance for X'"
  is NOT fine unless the user explicitly said that name.

STRUCTURE:
1. HOOK (1-2 lines): metric / bold statement / story opener / contrast
2. BODY (3-5 short paragraphs): context → what you did → result → takeaway
3. CLOSING: ONE specific question or CTA

STYLE: 140-250 words. Short paragraphs. 2-4 purposeful emojis.

BANNED: prestigious, renowned, world-class, synergy, leverage, game-changer,
spearheaded, incredible journey, passion for, hard work pays off,
"excited to share", "thrilled to announce", "I'm proud to", "humbled to",
"honoured to", "I am pleased to", "delighted to share"

OUTPUT: Post text only. No title, no label.\
"""


async def generator_node(state: GlobalState) -> dict:
    fact_list = _build_fact_list(state)
    try:
        response = await llm_write.ainvoke([
            SystemMessage(content=_GENERATOR_SYSTEM),
            HumanMessage(content=f"[FACTS]:\n{fact_list}\n\nWrite the LinkedIn post:"),
        ])
        draft_v1 = response.content.strip()
    except Exception as e:
        logger.error(f"[generator] LLM failed: {e}")
        draft_v1 = f"[Generator error: {e}]"
    logger.debug(f"[generator] draft length: {len(draft_v1)} chars")
    return {"_li_draft_v1": draft_v1, "current_agent": "Generator"}


# =============================================================================
# 4. Evaluator node
# =============================================================================

class EvalIssue(BaseModel):
    type: str = Field(description=(
        "invented_content|banned_phrase|cliche_hook|"
        "wall_of_text|weak_ending|wrong_length|other"
    ))
    description: str = Field(description="Specific issue.")


class EvalResult(BaseModel):
    has_issues: bool = Field(description="True if issues found.")
    issues: list[EvalIssue] = Field(default_factory=list)
    revised_post: str = Field(description="Fixed post (140-250 words). Copy if no issues.")


_EVALUATOR_SYSTEM = """\
Senior LinkedIn editor. Review draft vs fact list. Output EvalResult JSON.

CHECK:
A. invented_content — anything not traceable to the fact list
B. banned_phrase — prestigious/renowned/world-class/synergy/leverage/
   game-changer/spearheaded/"excited to share"/"thrilled to announce"/
   "I'm proud to"/"humbled to"/"honoured to"/"incredible journey"/
   "passion for"/"hard work pays off"/"I am pleased to"
C. cliche_hook — starts with excited/proud/humbled/thrilled/honoured
D. wall_of_text — paragraph > 3 lines without a break
E. weak_ending — no specific question or CTA
F. wrong_length — outside 140-250 words

REWRITE: Keep all facts. No new facts. Fix every issue.\
"""

_evaluator_structured = llm_precise.with_structured_output(EvalResult)


async def evaluator_node(state: GlobalState) -> dict:
    fact_list = _build_fact_list(state)
    draft_v1  = state.get("_li_draft_v1", "")
    try:
        result: EvalResult = await _evaluator_structured.ainvoke([
            SystemMessage(content=_EVALUATOR_SYSTEM),
            HumanMessage(content=f"[FACTS]:\n{fact_list}\n\n[DRAFT]:\n{draft_v1}\n\nOutput EvalResult JSON:"),
        ])
        if result.has_issues:
            logger.info(f"[evaluator] {len(result.issues)} issues: {[i.type for i in result.issues]}")
        else:
            logger.debug("[evaluator] no issues found")
        draft_v2 = result.revised_post.strip() or draft_v1
    except Exception as e:
        logger.warning(f"[evaluator] structured output failed ({e}) — keeping draft_v1")
        draft_v2 = draft_v1
    return {"_li_draft_v2": draft_v2, "current_agent": "Evaluator"}


# =============================================================================
# 5. Style matcher
# =============================================================================

_STYLE_MATCHER_SYSTEM = """\
Writing style expert. Rewrite the draft to match the user's voice from past posts.
Analyse: sentence length, vocabulary, emoji usage, hook style, closing style.
RULES: Do NOT change any facts. Keep 140-250 words.
OUTPUT: Final post only. No commentary.\
"""


async def style_matcher_node(state: GlobalState) -> dict:
    examples = state.get("_li_style_examples", "")
    draft_v2 = state.get("_li_draft_v2", "")

    if not examples or not examples.strip() or "No LinkedIn post" in examples:
        logger.debug("[style_matcher] no past posts — skipping")
        final_post = draft_v2
    else:
        try:
            response = await llm_write.ainvoke([
                SystemMessage(content=_STYLE_MATCHER_SYSTEM),
                HumanMessage(content=f"[PAST POSTS]:\n{examples}\n\n[DRAFT]:\n{draft_v2}\n\nRewrite:"),
            ])
            final_post = response.content.strip() or draft_v2
        except Exception as e:
            logger.warning(f"[style_matcher] failed ({e}) — using draft_v2")
            final_post = draft_v2

    return {
        "messages":       [AIMessage(content=final_post)],
        "_li_final_post": final_post,
        "current_agent":  "Style Matcher",
    }


# =============================================================================
# Subgraph assembly
# =============================================================================

_g = StateGraph(GlobalState)

_g.add_node("researcher",    researcher_node)
_g.add_node("clarifier",     clarifier_node)
_g.add_node("generator",     generator_node)
_g.add_node("evaluator",     evaluator_node)
_g.add_node("style_matcher", style_matcher_node)

_g.add_edge(START,        "researcher")
_g.add_edge("researcher", "clarifier")

_g.add_conditional_edges(
    "clarifier",
    _route_after_clarifier,
    {"generator": "generator", END: END},
)

_g.add_edge("generator",     "evaluator")
_g.add_edge("evaluator",     "style_matcher")
_g.add_edge("style_matcher", END)

linkedin_subgraph = _g.compile()
