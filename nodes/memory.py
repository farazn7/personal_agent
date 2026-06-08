"""
nodes/memory.py — memory_inject and memory_update graph nodes.

STM DESIGN:
  Keep last STM_WINDOW_SIZE messages verbatim in the checkpoint.
  Summarize older messages into stm_summary when count exceeds STM_SUMMARIZE_AFTER.
  After summarization, remove old messages from the checkpoint using RemoveMessage
  so the checkpoint doesn't grow unboundedly.

LTM DESIGN:
  Retrieve relevant facts from InMemoryStore (populated from PostgreSQL on startup).
  Extract new facts after each turn and write to both stores.
  With interrupt()-based HITL, no special bypass logic is needed — the graph
  only reaches memory_update after a full completion (including post-HITL).
"""

import re
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import RemoveMessage
from langgraph.store.base import BaseStore

from states.orchestrator import OrchestratorState
from shared import (
    last_human_content,
    last_ai_content,
    extract_keywords,
    is_trivial_message,
)
from config import (
    LTM_NAMESPACE_PREFIX, LTM_MAX_FACTS_PER_TURN,
    LTM_RETRIEVAL_LIMIT,
    STM_WINDOW_SIZE, STM_SUMMARIZE_AFTER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# memory_inject — runs BEFORE the router each turn
# =============================================================================

async def memory_inject_node(
    state: OrchestratorState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Retrieve relevant LTM facts and inject into state.
    """
    user_id    = state.get("user_id", "default")
    messages   = state.get("messages", [])
    user_input = last_human_content(messages)

    ltm_context = ""
    if store and user_input.strip() and not is_trivial_message(user_input):
        try:
            namespace = (LTM_NAMESPACE_PREFIX, user_id)
            kw_query  = " ".join(extract_keywords(user_input))
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

    return {"ltm_context": ltm_context}


# =============================================================================
# memory_update — runs AFTER every subgraph
# =============================================================================

def _should_extract_ltm(text: str, route: str) -> bool:
    """Decide whether to run LTM extraction for this turn."""
    if route == "linkedin":
        return True
    return not is_trivial_message(text)


async def memory_update_node(
    state: OrchestratorState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """
    Two jobs per turn:
      1. STM: summarize + trim messages older than the window
      2. LTM: extract new user facts and write to both stores
    """
    user_id  = state.get("user_id", "default")
    messages = state.get("messages", [])
    route    = state.get("route", "chat")

    user_input = last_human_content(messages)
    ai_output  = last_ai_content(messages)

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
    if store and user_input and _should_extract_ltm(user_input, route):
        try:
            from memory.ltm import extract_and_store_facts
            await extract_and_store_facts(
                user_input=user_input,
                ai_output=ai_output[:500],
                user_id=user_id,
                store=store,
            )
        except Exception as e:
            logger.warning(f"[memory_update] LTM extraction failed: {e}")

    return updates
