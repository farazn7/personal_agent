"""
tools/web_search.py — Async DuckDuckGo search wrapper.

No API key needed. Returns formatted text ready to inject into LLM context.
"""

import asyncio
import logging
from typing import List

from config import WEB_SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)


async def web_search_async(
    queries: List[str],
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> str:
    """
    Run one or more search queries concurrently.
    Returns a single formatted string of results, or empty string on failure.
    """
    if not queries:
        return ""

    loop = asyncio.get_running_loop()

    async def _search_one(query: str) -> str:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=max_results)),
            )
            if not results:
                return ""
            lines = [f"Query: {query}"]
            for r in results:
                title   = r.get("title", "")
                body    = r.get("body", "")[:300]
                href    = r.get("href", "")
                lines.append(f"  • {title}\n    {body}\n    Source: {href}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"[web_search] Query '{query[:40]}' failed: {e}")
            return ""

    # Run all queries concurrently
    tasks   = [_search_one(q) for q in queries]
    results = await asyncio.gather(*tasks)

    combined = "\n\n".join(r for r in results if r)
    if not combined:
        logger.debug("[web_search] No results returned.")
    return combined
