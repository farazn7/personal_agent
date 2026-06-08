"""
subgraphs/linkedin.py — LinkedIn post generation subgraph (isolated).

Pipeline:
  START → researcher → clarifier → generator → evaluator → style_matcher → END

State: LinkedInState (4 overlapping + 7 private fields)

HITL DESIGN — native interrupt():
  The clarifier calls interrupt(questions). The graph pauses mid-node and the
  checkpoint is saved. When the server calls Command(resume=answers), the
  interrupt() call returns the user's answers and the node continues.
  Up to 2 rounds. No state flags needed.
"""

import re
import json
import asyncio
import logging
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeInterrupt

from states.linkedin import LinkedInState
from shared import last_human_content, extract_keywords, build_rag_queries
from llm_config import llm_write, llm_precise
from tools.web_search import web_search_async
from tools.rag_engine import retrieve_linkedin_examples

logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================

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
# 1. Researcher node
# =============================================================================

async def researcher_node(state: LinkedInState) -> dict:
    """
    Concurrent: web search (if hybrid/open_book) + ChromaDB style examples.
    """
    messages       = state.get("messages", [])
    user_input     = last_human_content(messages)
    research_mode  = state.get("research_mode", "closed_book")
    search_queries = state.get("search_queries", [])
    queries        = search_queries or build_rag_queries(user_input)
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
        "web_results":    web_results,
        "style_examples": style_examples,
    }


# =============================================================================
# 2. Clarifier — multi-turn HITL via interrupt()
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


async def _evaluate_context(
    user_input: str,
    ltm_context: str,
    web_results: str,
    accumulated_answers: dict,
) -> tuple[bool, list[str]]:
    """
    Ask the LLM whether we have enough context to write the post.
    Returns (needed: bool, questions: list[str]).
    """
    known_facts = ltm_context.strip() or "(none)"
    web_context = web_results.strip() or "(no web research)"

    # Build context including accumulated answers (for re-evaluation on resume)
    answers_section = ""
    if accumulated_answers:
        lines = ["User's answers to previous questions:"]
        for q, a in accumulated_answers.items():
            answer_text = str(a).strip()
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

    return needed, questions


async def clarifier_node(state: LinkedInState) -> dict:
    """
    HITL clarifier using NodeInterrupt (Python 3.10 compatible).

    On first run: evaluates if questions are needed, raises NodeInterrupt
    with the questions to pause the graph.

    On resume: Command(resume=answers_dict) re-runs the node.
    The answers are available via state — the server passes them as
    hitl_answers in the Command update.
    """
    messages    = state.get("messages", [])
    user_input  = last_human_content(messages)
    ltm_context = state.get("ltm_context", "")
    web_results = state.get("web_results", "")

    # If we already have answers from a resume, we're done
    existing_answers = state.get("hitl_answers", {})
    if existing_answers:
        logger.debug(f"[clarifier] resumed with {len(existing_answers)} answers")
        return {"hitl_answers": existing_answers}

    # First run — check if we need to ask questions
    needed, questions = await _evaluate_context(
        user_input, ltm_context, web_results, {}
    )

    if not needed or not questions:
        logger.debug("[clarifier] sufficient context — no questions needed")
        return {"hitl_answers": {}}

    logger.info(f"[clarifier] HITL: {len(questions)} questions")

    # Pause the graph — NodeInterrupt works on Python 3.10
    raise NodeInterrupt({"questions": questions, "round": 1})


# =============================================================================
# Fact list builder
# =============================================================================

def _build_fact_list(state: LinkedInState) -> str:
    """
    Build the numbered fact list for the generator.
    Only includes confirmed, genuine information — no hallucination fodder.
    """
    items = []
    n     = 1

    messages   = state.get("messages", [])
    user_input = last_human_content(messages)

    if user_input.strip():
        items.append(f"{n}. [USER SAID]: {user_input.strip()}")
        n += 1

    hitl_answers = state.get("hitl_answers", {}) or {}
    for q, a in hitl_answers.items():
        answer_text = str(a).strip()
        if not answer_text or _is_non_answer(answer_text):
            logger.debug(f"[fact_list] skipping non-answer: '{answer_text[:40]}'")
            continue
        items.append(f"{n}. [USER CONFIRMED — {q[:60]}]: {answer_text}")
        n += 1

    web = state.get("web_results", "")
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


async def generator_node(state: LinkedInState) -> dict:
    fact_list = _build_fact_list(state)
    try:
        response = await llm_write.ainvoke([
            SystemMessage(content=_GENERATOR_SYSTEM),
            HumanMessage(content=f"[FACTS]:\n{fact_list}\n\nWrite the LinkedIn post:"),
        ])
        draft = response.content.strip()
    except Exception as e:
        logger.error(f"[generator] LLM failed: {e}")
        draft = f"[Generator error: {e}]"
    logger.debug(f"[generator] draft length: {len(draft)} chars")
    return {"draft": draft, "fact_list": fact_list}


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


async def evaluator_node(state: LinkedInState) -> dict:
    fact_list = _build_fact_list(state)
    draft     = state.get("draft", "")
    try:
        result: EvalResult = await _evaluator_structured.ainvoke([
            SystemMessage(content=_EVALUATOR_SYSTEM),
            HumanMessage(content=f"[FACTS]:\n{fact_list}\n\n[DRAFT]:\n{draft}\n\nOutput EvalResult JSON:"),
        ])
        if result.has_issues:
            logger.info(f"[evaluator] {len(result.issues)} issues: {[i.type for i in result.issues]}")
        else:
            logger.debug("[evaluator] no issues found")
        revised_draft = result.revised_post.strip() or draft
    except Exception as e:
        logger.warning(f"[evaluator] structured output failed ({e}) — keeping draft")
        revised_draft = draft
    return {"revised_draft": revised_draft}


# =============================================================================
# 5. Style matcher
# =============================================================================

_STYLE_MATCHER_SYSTEM = """\
Writing style expert. Rewrite the draft to match the user's voice from past posts.
Analyse: sentence length, vocabulary, emoji usage, hook style, closing style.
RULES: Do NOT change any facts. Keep 140-250 words.
OUTPUT: Final post only. No commentary.\
"""


async def style_matcher_node(state: LinkedInState) -> dict:
    examples      = state.get("style_examples", "")
    revised_draft = state.get("revised_draft", "")

    if not examples or not examples.strip() or "No LinkedIn post" in examples:
        logger.debug("[style_matcher] no past posts — skipping")
        final_post = revised_draft
    else:
        try:
            response = await llm_write.ainvoke([
                SystemMessage(content=_STYLE_MATCHER_SYSTEM),
                HumanMessage(content=f"[PAST POSTS]:\n{examples}\n\n[DRAFT]:\n{revised_draft}\n\nRewrite:"),
            ])
            final_post = response.content.strip() or revised_draft
        except Exception as e:
            logger.warning(f"[style_matcher] failed ({e}) — using revised draft")
            final_post = revised_draft

    return {
        "messages":   [AIMessage(content=final_post)],
        "final_post": final_post,
    }


# =============================================================================
# Subgraph factory
# =============================================================================

def build_linkedin_subgraph():
    """Build and compile the LinkedIn subgraph. Call this instead of using a module-level singleton."""
    g = StateGraph(LinkedInState)

    g.add_node("researcher",    researcher_node)
    g.add_node("clarifier",     clarifier_node)
    g.add_node("generator",     generator_node)
    g.add_node("evaluator",     evaluator_node)
    g.add_node("style_matcher", style_matcher_node)

    g.add_edge(START,            "researcher")
    g.add_edge("researcher",     "clarifier")
    g.add_edge("clarifier",      "generator")     # straight edge — interrupt() handles pause
    g.add_edge("generator",      "evaluator")
    g.add_edge("evaluator",      "style_matcher")
    g.add_edge("style_matcher",  END)

    return g.compile()
