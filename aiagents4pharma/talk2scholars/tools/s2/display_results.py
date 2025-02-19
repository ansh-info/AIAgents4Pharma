#!/usr/bin/env python3

"""
This tool is used to display the table of studies.
"""


import logging

from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""


@tool("display_results", parse_docstring=True)
def display_results(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """
    Display results after a search or recommendation.

    Args:
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        state (dict): The state of the agent containing the papers.

    Returns:
        str: A message indicating that the papers have been displayed.

    Raises:
        NoPapersFoundError: If no papers are found in the state.

    Note:
        The exception allows the LLM to make a more informed decision about initiating a search.
    """
    logger.info("Displaying papers")
    context_key = state.get("last_displayed_papers")
    artifact = state.get(context_key)
    if not artifact:
        logger.info("No papers found in state, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search/rec needs to be performed first."
        )
    content = f"{len(artifact)} papers found. Papers are attached as an artifact."
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=artifact,
                )
            ],
        }
    )
