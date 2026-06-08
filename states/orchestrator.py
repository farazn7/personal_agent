"""
OrchestratorState — the ONLY state the parent graph checkpoints.

Rule: if a field is not needed by memory, routing, or cross-subgraph
communication, it does NOT belong here.
"""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class OrchestratorState(TypedDict):
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    stm_summary: str
    ltm_context: str
    route: str                # "chat" | "rag" | "linkedin"
    research_mode: str        # "closed_book" | "hybrid" | "open_book"
    search_queries: list[str]
