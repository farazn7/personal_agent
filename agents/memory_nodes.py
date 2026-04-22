"""
agents/memory_nodes.py — memory_inject and memory_update graph nodes.

STM DESIGN:
  Keep last STM_WINDOW_SIZE messages verbatim in the checkpoint.
  Summarize older messages into stm_summary when count exceeds STM_SUMMARIZE_AFTER.
  After summarization, remove old messages from the checkpoint using RemoveMessage
  so the checkpoint doesn't grow unboundedly.

LTM DESIGN:
  Retrieve relevant facts from InMemoryStore (populated from PostgreSQL on startup).
  Extract new facts after each turn and write to both stores.
  HITL first-run: skip extraction (no output yet to extract from).
  HITL resume: DO extract — the user's detailed answers + generated post are valuable.
"""

import re
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.store.base import BaseStore

from state import GlobalState
from config import (
    LTM_NAMESPACE_PREFIX, LTM_MAX_FACTS_PER_TURN,
    LTM_RETRIEVAL_LIMIT,
    STM_WINDOW_SIZE, STM_SUMMARIZE_AFTER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

_LTM_SKIP_PATTERNS = [
    r"^(hi|hey|hello|good\s+morning|good\s+evening|good\s+night|sup|yo)[\s!.,]*$",
    r"^(ok|okay|thanks|thank\s+you|sure|yes|no|got\s+it|great|cool|nice|sounds\s+good)[\s!.,]*$",
    r"^(what|how|why|when|where|who|is|are|can|could|would|should|do|did)\b[^.!]*\?\s*$",
    r"\b(what\s+do\s+you\s+think|your\s+opinion|do\s+you\s+like|how\s+about\s+you)\b",
]

_KW_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "is","are","was","were","be","been","i","my","me","we","you","your","it","its",
    "what","can","could","would","should","have","has","had","do","did",
    "please","help","need","want","like","tell","show","give","find",
    "hi","hey","hello","ok","okay","yes","no","got","sure","this","that","which",
}


def _last_human_content(messages: list) -> str:
    """Return the content of the last HumanMessage. Never returns an AIMessage."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def _last_ai_content(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
    return ""


def _should_extract_ltm(text: str, route: str) -> bool:
    if route == "linkedin":
        return True
    if len(text.split()) < 3:
        return False
    return not any(re.search(p, text.strip(), re.IGNORECASE) for p in _LTM_SKIP_PATTERNS)


def _extract_keywords(text: str, max_kw: int = 6) -> list:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    seen, out = set(), []
    for w in cleaned.split():
        if w not in _KW_STOP and len(w) > 2 and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) >= max_kw:
                break
    return out


# =============================================================================
# memory_inject — runs BEFORE the router each turn
# =============================================================================

async def memory_inject_node(
    state: GlobalState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Retrieve relevant LTM facts and inject into state.

    HITL resume bypass: ltm_context is already in the checkpoint from the first
    run. Re-fetching would waste an embed call and return the same data.
    """
    if state.get("_li_hitl_complete", False):
        logger.debug("[memory_inject] HITL resume — reusing existing ltm_context")
        return {"current_agent": "Memory"}

    user_id    = state.get("user_id", "default")
    messages   = state.get("messages", [])
    user_input = _last_human_content(messages)

    ltm_context = ""
    if store and user_input.strip():
        trivial = (
            len(user_input.split()) <= 2
            or any(re.search(p, user_input.strip(), re.IGNORECASE) for p in _LTM_SKIP_PATTERNS)
        )
        if not trivial:
            try:
                namespace = (LTM_NAMESPACE_PREFIX, user_id)
                kw_query  = " ".join(_extract_keywords(user_input))
                results   = await store.asearch(
                    namespace, query=kw_query, limit=LTM_RETRIEVAL_LIMIT,
                )
                if results:
                    lines = ["[Known facts about you]:"]
                    for item in results:
                        fact = item.value.get("fact", "")
                        cat  = item.value.get("category", "")
                        if fact:
                            lines.append(f"  • [{cat}] {fact}")
                    ltm_context = "\n".join(lines)
                    logger.debug(f"[memory_inject] retrieved {len(results)} LTM facts")
            except Exception as e:
                logger.warning(f"[memory_inject] LTM retrieval failed: {e}")

    return {"ltm_context": ltm_context, "current_agent": "Memory"}


# =============================================================================
# memory_update — runs AFTER every subgraph
# =============================================================================

async def memory_update_node(
    state: GlobalState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Two jobs per turn:
      1. STM: summarize + trim messages older than the window
      2. LTM: extract new user facts and write to both stores

    HITL FIRST RUN (needs_hitl=True, complete=False):
      Skip LTM extraction — there's no generated output to extract from yet.
      The pipeline stopped at the clarifier; no post was written.

    HITL RESUME (complete=True):
      DO run LTM extraction. The user gave detailed answers and a full post
      was generated. Both contain high-value extractable facts.
    """
    is_hitl_first_run = (
        state.get("_li_needs_hitl", False) and not state.get("_li_hitl_complete", False)
    )

    user_id  = state.get("user_id", "default")
    messages = state.get("messages", [])
    route    = state.get("route", "chat")

    user_input = _last_human_content(messages)
    ai_output  = _last_ai_content(messages)

    updates: dict = {}

    # ── 1. STM: summarize old messages and trim checkpoint ─────────────────
    # Only trigger when there are messages genuinely outside the window AND
    # we're past the summary threshold (avoids summarizing every single turn).
    num_outside_window = len(messages) - STM_WINDOW_SIZE
    if num_outside_window > 0 and len(messages) > STM_SUMMARIZE_AFTER:
        try:
            from memory.stm import summarize_old_messages
            new_summary, recent = await summarize_old_messages(
                messages=messages,
                existing_summary=state.get("stm_summary", ""),
                window_size=STM_WINDOW_SIZE,
            )
            updates["stm_summary"] = new_summary

            # Remove old messages from checkpoint using RemoveMessage.
            # The add_messages reducer processes RemoveMessage ops to delete entries.
            # This prevents the checkpoint from growing unboundedly.
            old_messages = messages[:-STM_WINDOW_SIZE]
            remove_ops = [
                RemoveMessage(id=m.id)
                for m in old_messages
                if hasattr(m, "id") and m.id
            ]
            if remove_ops:
                updates["messages"] = remove_ops
                logger.info(
                    f"[memory_update] STM: summarized {len(old_messages)} msgs, "
                    f"trimmed {len(remove_ops)}, kept {len(recent)}"
                )
        except Exception as e:
            logger.warning(f"[memory_update] STM failed: {e}")

    # ── 2. LTM: extract facts ──────────────────────────────────────────────
    if is_hitl_first_run:
        logger.debug("[memory_update] HITL first run — skipping LTM (no output yet)")
    elif store and user_input and _should_extract_ltm(user_input, route):
        try:
            from memory.ltm import extract_and_store_facts

            # On HITL resume: include the HITL answers in the extraction context
            # so the user's detailed clarifier answers are stored as facts
            extract_input = user_input
            if state.get("_li_hitl_complete", False):
                hitl_answers = state.get("_li_hitl_answers", {}) or {}
                if hitl_answers:
                    answers_text = "; ".join(f"{a}" for a in hitl_answers.values() if str(a).strip())
                    if answers_text:
                        extract_input = f"{user_input}. Additional context: {answers_text}"

            await extract_and_store_facts(
                user_input=extract_input,
                ai_output=ai_output[:500],
                user_id=user_id,
                store=store,
            )
        except Exception as e:
            logger.warning(f"[memory_update] LTM extraction failed: {e}")

    return updates
