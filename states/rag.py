"""RAGState — overlapping + 2 private fields for retrieval results."""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ltm_context: str
    research_mode: str
    search_queries: list[str]
    # Private
    chroma_results: str
    web_results: str
