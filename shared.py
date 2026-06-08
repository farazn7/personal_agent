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
