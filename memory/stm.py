"""
memory/stm.py — Short-Term Memory helpers.

Strategy (from reference 5_stm_summarization.ipynb):
  - Keep last STM_WINDOW_SIZE messages verbatim (immediacy)
  - Summarize everything older into a rolling summary string
  - Summary + recent messages = what the LLM sees each turn

The summarizer runs as an async function called from memory_update_node.
It wraps the sync Ollama call in run_in_executor so it never blocks the loop.
"""

import asyncio
import logging
from typing import List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from llm_config import llm_summarizer

logger = logging.getLogger(__name__)

_SUMMARIZER_SYSTEM = """\
You are a conversation summarizer for a personal AI assistant.

Given a conversation (and optionally a prior summary), produce a concise
updated summary in 3-6 bullet points.

Focus ONLY on:
  • Factual information the user shared about themselves
  • Topics discussed and key conclusions reached
  • Decisions made or tasks completed
  • Any context needed to understand future messages

Skip: pleasantries, filler, assistant explanations, questions that
were answered (keep only the answer).

If there is a prior summary, extend it with new information.
Do NOT repeat what the prior summary already contains.

Output ONLY the bullet list — no preamble, no labels, no headers.\
"""


def _messages_to_text(messages: List[BaseMessage]) -> str:
    """Format messages as plain text for the summarizer."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Cap long AI responses (LinkedIn posts etc.) to avoid bloating prompt
            content = msg.content[:500] + "…" if len(msg.content) > 500 else msg.content
            lines.append(f"Assistant: {content}")
        # Skip SystemMessages — they're injected context, not conversation
    return "\n".join(lines)


def _summarize_sync(messages: List[BaseMessage], existing_summary: str) -> str:
    """Synchronous summarization — only called via run_in_executor."""
    if not messages:
        return existing_summary

    conversation_text = _messages_to_text(messages)
    prompt_parts = []

    if existing_summary:
        prompt_parts.append(f"Prior summary:\n{existing_summary}\n")

    prompt_parts.append(f"New conversation to incorporate:\n{conversation_text}")

    try:
        response = llm_summarizer.invoke([
            SystemMessage(content=_SUMMARIZER_SYSTEM),
            HumanMessage(content="\n\n".join(prompt_parts)),
        ])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[stm] Summarization LLM failed: {e}")
        # Fallback: return last 800 chars of raw text
        return conversation_text[-800:]


async def summarize_old_messages(
    messages: List[BaseMessage],
    existing_summary: str,
    window_size: int,
) -> Tuple[str, List[BaseMessage]]:
    """
    Async-safe STM trimmer.

    Returns (new_summary, recent_messages) where recent_messages are the
    last window_size messages to keep verbatim.

    The sync LLM call runs in run_in_executor so it doesn't block the event loop.
    """
    if len(messages) <= window_size:
        return existing_summary, messages

    older  = messages[:-window_size]
    recent = messages[-window_size:]

    loop = asyncio.get_running_loop()
    new_summary = await loop.run_in_executor(
        None,
        _summarize_sync,
        older,
        existing_summary,
    )

    logger.info(f"[stm] Summarized {len(older)} old messages → kept {len(recent)} recent")
    return new_summary, recent


def build_llm_messages(
    recent_messages: List[BaseMessage],
    summary: str,
    system_prompt: str,
    ltm_context: str = "",
    max_messages: int = 12,
) -> List[BaseMessage]:
    """
    Build the final message list to pass to an LLM call.

    Order:
      [SystemMessage: system_prompt]
      [SystemMessage: stm_summary]       ← if exists
      [SystemMessage: ltm_context]       ← if exists
      [recent messages... (capped at max_messages)]

    max_messages hard cap: safety net so callers that forget to trim
    never flood Ollama with hundreds of messages.
    """
    result: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    if summary:
        result.append(SystemMessage(
            content=f"[Context from earlier in this conversation]:\n{summary}"
        ))

    if ltm_context:
        result.append(SystemMessage(content=ltm_context))

    # Safety cap — never send more than max_messages to avoid context overflow
    capped = recent_messages[-max_messages:] if len(recent_messages) > max_messages else recent_messages
    result.extend(capped)
    return result
