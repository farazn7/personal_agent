"""
subgraphs/chat.py — Chat subgraph.

Pipeline: START → chatbot_node → END

FIXES APPLIED:
  1. Message trimming — only last STM_WINDOW_SIZE messages go to Ollama.
     Passing full history was causing 26s responses and context bleed
     where the LLM "remembered" previous topics and mixed them with
     current answers.

  2. Follow-up web search detection — if the previous AI message offered
     to search ("want me to search", "should I look this up") and the
     user replied affirmatively ("yes", "yes please", "go ahead"), the
     chatbot automatically triggers a web search using the topic extracted
     from the previous AI message.

  3. Stronger grounding when web results present — explicit instruction
     to ONLY use web result content, never training data for the topic.
     This prevents the LLM from fabricating citations like "BBC News [1]"
     from its training knowledge.

  4. Web search query uses STM context for pronoun resolution
     (e.g. "what about it?" → extracts proper nouns from summary).
"""

import re
import logging
from datetime import date
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from state import GlobalState
from config import STM_WINDOW_SIZE
from llm_config import llm_write as llm_fast
from memory.stm import build_llm_messages
from tools.web_search import web_search_async

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Web search trigger logic
# ══════════════════════════════════════════════════════════════════════════

# Skip web search — checked FIRST (short-circuit)
_WEB_SKIP = [
    # Pure greetings / one-word acks
    r"^(hi|hey|hello|ok|okay|thanks|thank\s*you|sure|no|got\s*it|great|cool|nice)[\s!.,]*$",
    # Definitional / evergreen questions
    r"\bwhat\s+(is|are)\s+(a|an|the)\s+\w+\b(?!\s*(doing|happening|situation|update|status))",
    r"\bexplain\b.{0,40}\b(concept|theorem|algorithm|formula|equation|difference|how)\b",
    r"\bdifference\s+between\b",
    # Coding / implementation
    r"\b(code|implement|write\s+a\s+function|debug|script|program|error\s+in)\b",
    # About the assistant itself
    r"\b(who\s+are\s+you|what\s+can\s+you\s+do|how\s+do\s+you\s+work|are\s+you\s+an\s+ai)\b",
]

# Trigger web search
_WEB_TRIGGERS = [
    # Year references
    r"\b(202[3-9]|20[3-9]\d)\b",
    # Recency signals
    r"\b(latest|recent|current|today|this\s+week|last\s+week|this\s+year|last\s+year|now|live|breaking|right\s+now|just\s+happened)\b",
    # Factual current-state questions
    r"\b(who\s+is|who\s+was|where\s+is|when\s+is|when\s+did|is\s+it\s+still)\b",
    # Current events topics
    r"\b(news|update|updates|score|result|results|price|prices|schedule|standings|winner|champion|election|war|conflict|attack|crisis|disaster|earthquake|flood|hurricane)\b",
    r"\bwhat\s+happened\b",
    r"\bstill\s+(happening|running|active|available|open|going|alive|on)\b",
    r"\b(stock|crypto|bitcoin|market|exchange\s+rate|inflation|gdp)\b",
    r"\bweather\s+(in|at|for|today|tomorrow|tonight)\b",
    r"\bhow\s+much\s+(does|is|are|did|cost)\b",
    # Status / existence
    r"\b(is|are)\s+.{1,30}\s+(still|currently|happening|active|available|open)\b",
]

# Affirmative follow-up patterns (for detecting "yes please search")
_AFFIRMATIVE = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|go\s+ahead|please|do\s+it|please\s+do|"
    r"yes\s+please|yes\s+do|do\s+that|go\s+for\s+it|absolutely|definitely|"
    r"yes\s+search|search\s+for\s+it|search\s+it|look\s+it\s+up|find\s+out)",
    re.IGNORECASE,
)

# Signals that previous AI message offered a web search
_SEARCH_OFFER = re.compile(
    r"(want\s+me\s+to\s+search|should\s+i\s+search|shall\s+i\s+look|"
    r"i\s+can\s+search|i\s+could\s+search|i\s+don.t\s+have\s+live|"
    r"i\s+don.t\s+have\s+(real.time|current|live|up.to.date)|"
    r"want\s+me\s+to\s+look|let\s+me\s+search|i\s+can\s+look\s+that\s+up)",
    re.IGNORECASE,
)


def _needs_web(user_input: str, prev_ai_msg: str = "") -> tuple[bool, str]:
    """
    Returns (should_search: bool, search_query: str).

    Two cases trigger web search:
      1. Direct: current message matches trigger patterns
      2. Follow-up: user affirms a previous AI offer to search
         → extract topic from the previous AI message
    """
    text = user_input.strip()

    # Case 2: Follow-up confirmation
    if prev_ai_msg and _AFFIRMATIVE.match(text) and _SEARCH_OFFER.search(prev_ai_msg):
        # Extract the topic the previous AI was offering to search
        topic = _extract_search_topic(prev_ai_msg, user_input)
        if topic:
            logger.debug(f"[chat] follow-up search confirmed → topic: {topic[:50]}")
            return True, topic

    # Skip list — checked before triggers
    if any(re.search(p, text, re.IGNORECASE) for p in _WEB_SKIP):
        return False, ""

    # Trigger list
    if any(re.search(p, text, re.IGNORECASE) for p in _WEB_TRIGGERS):
        return True, text

    return False, ""


def _extract_search_topic(ai_msg: str, user_msg: str) -> str:
    """
    Extract the topic the AI was offering to search from its message.
    e.g. "I don't have live info on the Iran-US situation. Want me to search?"
    → "Iran US situation"
    """
    # Remove the offer part and extract the subject
    cleaned = re.sub(
        r"(want me to search.*|shall i.*|i can search.*|let me search.*|"
        r"i don.t have live.*?\.)\s*$",
        "",
        ai_msg,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Also use any extra keywords from the user's confirmation
    extra = re.sub(r"^(yes|yeah|ok|sure|please|go ahead|do it)\s*", "", user_msg, flags=re.IGNORECASE).strip()

    # Combine: topic from AI + extra keywords from user
    parts = []
    if cleaned and len(cleaned) > 10:
        parts.append(cleaned[-200:])  # last 200 chars of AI context
    if extra and len(extra) > 3 and extra.lower() not in ("and tell", "tell me", "please"):
        parts.append(extra)

    return " ".join(parts).strip() if parts else cleaned[:150]


def _build_search_query(user_input: str, stm_summary: str) -> str:
    """
    Build a context-enriched search query.
    Resolves ambiguous pronouns using proper nouns from STM summary.
    """
    # Strip filler prefixes
    query = re.sub(
        r"^(can\s+you|could\s+you|please|tell\s+me|help\s+me\s+(understand|with))\s+",
        "",
        user_input.strip(),
        flags=re.IGNORECASE,
    ).strip() or user_input.strip()

    _AMBIGUOUS = re.compile(
        r"\b(their|his|her|its|them|they|the\s+war|the\s+conflict|"
        r"the\s+situation|the\s+crisis|the\s+attack|the\s+election|"
        r"the\s+match|the\s+game|the\s+tournament|it|this)\b",
        re.IGNORECASE,
    )

    if _AMBIGUOUS.search(query) and stm_summary:
        _COMMON = {
            "The","In","At","On","For","From","And","But","With",
            "This","That","It","He","She","They","Was","Is","Are",
        }
        nouns = re.findall(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", stm_summary)
        nouns = [n for n in nouns if n not in _COMMON][:3]
        if nouns:
            query = " ".join(nouns) + " " + query

    return query.strip()


def _result_is_relevant(web_text: str, user_input: str) -> bool:
    """Only inject web results if they share keywords with the question."""
    if not web_text:
        return False
    _STOP = {
        "what","when","where","which","that","this","they","them","with","from",
        "have","been","into","your","about","would","could","should","does",
        "how","why","who","the","and","or","yes","please","search",
    }
    words = [
        w.lower() for w in re.findall(r"\b\w{3,}\b", user_input)
        if w.lower() not in _STOP
    ]
    if not words:
        return True
    ctx = web_text.lower()
    return any(w in ctx for w in words[:8])


# ══════════════════════════════════════════════════════════════════════════
# System prompts
# ══════════════════════════════════════════════════════════════════════════

_BASE_SYSTEM = """\
You are a friendly, direct, and knowledgeable personal AI assistant.

IDENTITY — NON-NEGOTIABLE:
- You are an AI. You have NOT written code, worked at companies, built projects,
  or had personal experiences. Never say "I've worked with X" or "I built".

GREETINGS: Respond warmly and briefly. "Hey! What can I help you with?"
Never redirect a greeting to a database search.

NO WEB RESULTS — answer from training knowledge when confident.
For current events / live data you don't know: say "I don't have live info
on [X]. Want me to search the web?" — do NOT invent an answer.

PERSONAL PROFILE QUESTIONS: Only redirect to database for EXPLICIT queries
like "what skills do I have" or "search my resume".

Keep responses concise. No preamble. No "Great question!".\
"""

# Stricter version injected when web results are present
_WEB_GROUNDING_SYSTEM = """\
CRITICAL — WEB RESULTS GROUNDING:
Web search results are provided below. You MUST follow these rules EXACTLY:

1. Answer ONLY using the content in the web results.
2. Do NOT use any knowledge from your training about this specific topic.
   Your training data may be outdated or wrong — the search results are authoritative.
3. Do NOT fabricate, invent, or hallucinate ANY citation, URL, source name,
   or fact that is not explicitly present in the results below.
4. If the web results don't answer the question, say:
   "The search results don't contain clear information about this. Want me to try a different search?"
5. Do NOT add context from 2020, 2021, 2022, 2023 etc. unless the results mention it.
   Only report what the results say about the current situation.

Attribute facts naturally: "According to search results..." — never invent source names.\
"""


# ══════════════════════════════════════════════════════════════════════════
# Chatbot node
# ══════════════════════════════════════════════════════════════════════════

async def chatbot_node(state: GlobalState) -> dict:
    """
    Chat node with message trimming, follow-up web search, and strict grounding.
    """
    all_messages = state.get("messages", [])
    stm_summary  = state.get("stm_summary", "")
    ltm_context  = state.get("ltm_context", "")

    # Always use the last HumanMessage — never accidentally use an AIMessage
    user_input = ""
    for m in reversed(all_messages):
        if isinstance(m, HumanMessage):
            user_input = m.content
            break

    # Get the previous AI message for follow-up detection
    ai_msgs = [m for m in all_messages if isinstance(m, AIMessage)]
    prev_ai = ai_msgs[-1].content if ai_msgs else ""

    # ── CRITICAL: Trim messages to window size ─────────────────────────────
    # Do NOT pass the full history to Ollama. With 20+ turns this causes:
    #   - 26s+ inference time
    #   - Context window overflow
    #   - LLM mixing previous topics into current answer
    recent_messages = all_messages[-STM_WINDOW_SIZE:] if len(all_messages) > STM_WINDOW_SIZE else all_messages

    # ── Web search ─────────────────────────────────────────────────────────
    web_context = ""
    should_search, raw_query = _needs_web(user_input, prev_ai)

    if should_search:
        # Enrich query with STM context for pronoun resolution
        query = _build_search_query(raw_query, stm_summary)
        logger.info(f"[chat] web search → {query[:70]}")
        try:
            raw_web = await web_search_async([query], max_results=5)
            if raw_web and "no results" not in raw_web.lower():
                if _result_is_relevant(raw_web, user_input) or _result_is_relevant(raw_web, raw_query):
                    web_context = raw_web
                    logger.debug("[chat] web results accepted")
                else:
                    logger.debug("[chat] web results not relevant — discarded")
        except Exception as e:
            logger.warning(f"[chat] web search failed: {e}")

    # ── Build LLM message list ─────────────────────────────────────────────
    # Use trimmed recent messages only — NOT full history
    system_prompt = _BASE_SYSTEM
    llm_messages  = build_llm_messages(
        recent_messages=recent_messages,
        summary=stm_summary,
        system_prompt=system_prompt,
        ltm_context=ltm_context,
    )

    if web_context:
        # Insert BOTH grounding instruction and results before the last human message
        llm_messages.insert(-1, SystemMessage(content=_WEB_GROUNDING_SYSTEM))
        llm_messages.insert(-1, SystemMessage(
            content=f"[WEB SEARCH RESULTS]:\n{web_context}"
        ))

    # ── LLM call ───────────────────────────────────────────────────────────
    try:
        response = await llm_fast.ainvoke(llm_messages)
        reply    = response.content.strip()
    except Exception as e:
        logger.error(f"[chat] LLM failed: {e}")
        reply = f"Error: {e}\n\nIs Ollama running? Try: `ollama serve`"

    return {
        "messages":      [AIMessage(content=reply)],
        "current_agent": "Chatbot",
    }


# ══════════════════════════════════════════════════════════════════════════
# Subgraph assembly
# ══════════════════════════════════════════════════════════════════════════

_g = StateGraph(GlobalState)
_g.add_node("chatbot", chatbot_node)
_g.add_edge(START, "chatbot")
_g.add_edge("chatbot", END)

chat_subgraph = _g.compile()
