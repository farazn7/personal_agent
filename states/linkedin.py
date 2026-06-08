"""LinkedInState — overlapping + private fields for the generation pipeline."""
from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class LinkedInState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ltm_context: str
    research_mode: str
    search_queries: list[str]
    # Private: Research
    web_results: str
    style_examples: str
    # Private: Generation
    fact_list: str
    draft: str
    revised_draft: str
    final_post: str
    # Private: HITL
    hitl_answers: dict
