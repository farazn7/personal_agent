"""
memory/ltm.py — Long-Term Memory with dual persistence.

InMemoryStore  — fast in-process semantic search (LangGraph native)
PostgreSQL     — persistent storage that survives server restarts

On startup: load_ltm_from_postgres() populates the InMemoryStore from
PostgreSQL so all previously learned facts are immediately available.

On extraction: every new fact is written to BOTH stores atomically.
On deletion: removed from both stores.
"""

import uuid
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.store.base import BaseStore
from langsmith import traceable

from config import (
    LTM_MAX_FACTS_PER_TURN, LTM_NAMESPACE_PREFIX,
    PG_CONN_STRING,
)
from llm_config import llm_precise

logger = logging.getLogger(__name__)


# =============================================================================
# Structured extraction schemas
# =============================================================================

class MemoryItem(BaseModel):
    category: str = Field(
        description=(
            "One of: skill, achievement, project, event, role, "
            "education, preference, identity"
        )
    )
    fact: str = Field(
        description="Complete, self-contained atomic sentence about the user."
    )
    is_new: bool = Field(
        description=(
            "True if this fact is NEW — not already in existing memories. "
            "False if it's a duplicate or rephrasing of something already known."
        )
    )
    confidence: int = Field(
        description=(
            "1 = mentioned casually, "
            "2 = mentioned with detail/context, "
            "3 = stated directly and explicitly."
        )
    )


class MemoryDecision(BaseModel):
    should_write: bool = Field(
        description="True if any facts worth storing were found."
    )
    memories: List[MemoryItem] = Field(default_factory=list)


# =============================================================================
# Extraction prompt
# =============================================================================

_EXTRACT_SYSTEM = """\
You are extracting long-term personal facts about the USER from a conversation.

CRITICAL RULE: Every fact MUST be a complete, self-contained sentence.
Someone reading the fact with NO other context should fully understand it.
Include WHO, WHAT, WHERE, WHEN if mentioned. Include project/event names.

CATEGORIES:
  skill        — Named technical/professional skill they actually use.
                 GOOD: "Proficient in Python and PyTorch for ML projects"
                 BAD:  "Likes technology" (vague)
  achievement  — Concrete accomplishment with measurable outcome.
                 GOOD: "Won first place at the IRC robotics competition 2025"
                 BAD:  "Did well" (no specifics)
  project      — Specific project they built. MUST include project name/description + tech used.
                 GOOD: "Built a home security system using agentic AI with per-camera agents and spatial-temporal anomaly detection"
                 BAD:  "Used confidence levels to detect suspicious frames" (fragment, no project context)
  event        — Named competition, hackathon, or conference attended.
                 GOOD: "Participated in a CN competition building an agentic AI architecture for suspicious activity detection"
                 BAD:  "Participated in a competition" (no name or topic)
  role         — Specific job title or organizational role.
  education    — Degree, major, institution — only if explicitly stated.
  preference   — Strong, explicitly stated work or career preference.
  identity     — Name, city, current employer — only if directly stated.

IMPORTANT:
  - PREFER fewer, richer facts over many fragments.
  - Combine related details into ONE fact instead of splitting.
    BAD:  3 separate facts about the same project ("used spatial data", "used agents", "detected anomalies")
    GOOD: 1 combined fact ("Built a project using agentic AI with spatial-temporal data to detect anomalies in home security cameras")
  - Do NOT extract fragments that only make sense in the conversation context.
  - Do NOT extract anything the ASSISTANT said — only USER facts.
  - Do NOT extract future plans, questions, or transient states.

EXISTING MEMORIES (do not re-extract):
{existing_memories}

Set is_new=false if already covered by existing memories above.
Return should_write=false if nothing qualifies.\
"""

_extractor = llm_precise.with_structured_output(MemoryDecision)


# =============================================================================
# PostgreSQL helpers
# =============================================================================

async def _pg_write_fact(
    user_id: str,
    key: str,
    fact: str,
    category: str,
    confidence: int,
) -> None:
    """Write a single fact to PostgreSQL. Uses INSERT ON CONFLICT DO UPDATE."""
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(
            PG_CONN_STRING, connect_timeout=3
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO ltm_facts (user_id, category, fact, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, fact) DO UPDATE
                        SET confidence  = GREATEST(ltm_facts.confidence, EXCLUDED.confidence),
                            updated_at  = NOW(),
                            access_count = ltm_facts.access_count + 1
                """, (user_id, category, fact, confidence))
            await conn.commit()
    except Exception as e:
        logger.warning(f"[ltm] PostgreSQL write failed (fact still in memory): {e}")


async def _pg_delete_fact(user_id: str, fact_text: str) -> None:
    """Delete a fact from PostgreSQL by exact fact text."""
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(
            PG_CONN_STRING, connect_timeout=3
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM ltm_facts WHERE user_id = %s AND fact = %s",
                    (user_id, fact_text),
                )
            await conn.commit()
    except Exception as e:
        logger.warning(f"[ltm] PostgreSQL delete failed: {e}")


async def load_ltm_from_postgres(user_id: str, store: BaseStore) -> int:
    """
    Load all LTM facts from PostgreSQL into the InMemoryStore.
    Called once at server startup so facts survive restarts.
    Returns number of facts loaded.
    """
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(
            PG_CONN_STRING, connect_timeout=3
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT fact, category, confidence
                    FROM ltm_facts
                    WHERE user_id = %s
                    ORDER BY confidence DESC, access_count DESC
                """, (user_id,))
                rows = await cur.fetchall()

        if not rows:
            logger.info(f"[ltm] No facts in PostgreSQL for user '{user_id}'")
            return 0

        namespace = (LTM_NAMESPACE_PREFIX, user_id)
        loaded = 0
        for fact, category, confidence in rows:
            try:
                key = str(uuid.uuid4())
                await store.aput(namespace, key, {
                    "fact":       fact,
                    "category":   category,
                    "confidence": confidence,
                })
                loaded += 1
            except Exception as e:
                logger.warning(f"[ltm] Failed to load fact into store: {e}")

        logger.info(f"[ltm] Loaded {loaded} facts from PostgreSQL into memory for '{user_id}'")
        return loaded

    except Exception as e:
        logger.warning(f"[ltm] Could not load from PostgreSQL ({e}) — starting with empty memory")
        return 0


# =============================================================================
# Extraction — called from memory_update_node
# =============================================================================

@traceable(run_type="tool", name="ltm_extract_facts")
async def extract_and_store_facts(
    user_input: str,
    ai_output: str,
    user_id: str,
    store: BaseStore,
) -> int:
    """
    Extract new facts from a conversation turn.
    Writes to BOTH InMemoryStore (fast retrieval) AND PostgreSQL (persistence).
    Returns number of new facts stored.
    """
    namespace = (LTM_NAMESPACE_PREFIX, user_id)

    # Load existing memories for deduplication
    try:
        existing_items = await store.asearch(namespace, query=user_input, limit=20)
        existing_texts = [
            f"  • [{it.value.get('category','')}] {it.value.get('fact','')}"
            for it in existing_items
            if it.value.get("fact")
        ]
        existing_str = "\n".join(existing_texts) if existing_texts else "(none yet)"
    except Exception as e:
        logger.warning(f"[ltm] Could not load existing memories: {e}")
        existing_str = "(unknown)"

    conversation = f"User: {user_input}\nAssistant: {ai_output}"

    try:
        decision: MemoryDecision = await _extractor.ainvoke([
            SystemMessage(content=_EXTRACT_SYSTEM.format(existing_memories=existing_str)),
            HumanMessage(content=f"Conversation:\n\n{conversation}"),
        ])
    except Exception as e:
        logger.warning(f"[ltm] Extraction LLM failed: {e}")
        return 0

    if not decision.should_write or not decision.memories:
        logger.debug("[ltm] Nothing worth storing.")
        return 0

    stored = 0
    for mem in decision.memories[:LTM_MAX_FACTS_PER_TURN]:
        if not mem.is_new:
            logger.debug(f"[ltm] Skipping duplicate: {mem.fact[:60]}")
            continue
        if mem.confidence < 1:
            continue
        try:
            key = str(uuid.uuid4())
            # Write to InMemoryStore
            await store.aput(namespace, key, {
                "fact":       mem.fact,
                "category":   mem.category,
                "confidence": mem.confidence,
            })
            # Write to PostgreSQL (survives restart)
            await _pg_write_fact(user_id, key, mem.fact, mem.category, mem.confidence)
            stored += 1
            logger.info(f"[ltm] Stored [{mem.category}] conf={mem.confidence}: {mem.fact[:70]}")
        except Exception as e:
            logger.warning(f"[ltm] Store write failed: {e}")

    return stored


# =============================================================================
# Retrieval helpers for UI / server.py
# =============================================================================

async def get_all_ltm_facts(user_id: str, store: BaseStore, limit: int = 100) -> list:
    """Return all LTM facts for UI display. Reads from InMemoryStore (mirrors PostgreSQL)."""
    try:
        namespace = (LTM_NAMESPACE_PREFIX, user_id)
        items = await store.asearch(namespace, query="", limit=limit)
        return [
            {
                "key":        item.key,
                "fact":       item.value.get("fact", ""),
                "category":   item.value.get("category", ""),
                "confidence": item.value.get("confidence", 1),
            }
            for item in items
            if item.value.get("fact")
        ]
    except Exception as e:
        logger.error(f"[ltm] get_all_ltm_facts failed: {e}")
        return []


async def delete_ltm_fact(user_id: str, key: str, store: BaseStore) -> bool:
    """Delete a fact from both InMemoryStore and PostgreSQL."""
    try:
        namespace = (LTM_NAMESPACE_PREFIX, user_id)
        # Get fact text before deleting (needed for PostgreSQL delete)
        try:
            item = await store.aget(namespace, key)
            fact_text = item.value.get("fact", "") if item else ""
        except Exception:
            fact_text = ""
        # Delete from InMemoryStore
        await store.adelete(namespace, key)
        # Delete from PostgreSQL
        if fact_text:
            await _pg_delete_fact(user_id, fact_text)
        return True
    except Exception as e:
        logger.error(f"[ltm] delete failed: {e}")
        return False
