"""
state.py — GlobalState definition.
"""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class GlobalState(TypedDict):
    # Identity
    user_id: str

    # Conversation — add_messages reducer appends, never replaces
    messages: Annotated[list[BaseMessage], add_messages]

    # STM — rolling summary of messages older than the window
    stm_summary: str

    # Routing
    route: str            # "chat" | "linkedin" | "rag"
    research_mode: str    # "closed_book" | "hybrid" | "open_book"
    search_queries: list

    # Memory context injected by memory_inject_node
    ltm_context: str

    # Pipeline progress label
    current_agent: str

    # LinkedIn fields (reset each fresh turn in _build_initial_state)
    _li_web_results:    str
    _li_style_examples: str
    _li_hitl_answers:   dict   # accumulated answers across all HITL rounds
    _li_draft_v1:       str
    _li_draft_v2:       str
    _li_final_post:     str
    # State-flag HITL
    _li_needs_hitl:     bool   # clarifier wants more info — route to END
    _li_hitl_questions: list   # questions to show the user
    _li_hitl_complete:  bool   # user submitted answers — skip researcher
    _li_hitl_rounds:    int    # how many HITL rounds done (cap at 2)

    # RAG fields
    _rag_chroma_facts: str
    _rag_web_results:  str
