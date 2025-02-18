#!/usr/bin/env python3

"""
This tool is used to display the table of studies.
"""

import logging
from typing import Annotated, Literal
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""

@tool("display_results")
def display_results(
    context: Literal["search", "single_paper_rec", "multi_paper_rec"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]) -> str:
    """
    Display the papers in the state. If no papers are found, raises an exception
    indicating that a search is needed.

    Args:
        context (str): The context in which the tool is called.
        state (dict): The state of the agent containing the papers.

    Returns:
        str: A message indicating that the papers have been displayed.

    Raises:
        NoPapersFoundError: If no papers are found in the state.

    Note:
        The exception allows the LLM to make a more informed decision about initiating a search.
    """
    logger.info("Displaying papers with context: %s", context)

    if context == "search" or context == "single_paper_rec":
        if not state.get("papers") and not state.get("multi_papers"):
            logger.info("No papers found in state, raising NoPapersFoundError")
            raise NoPapersFoundError(
                "No papers found. A search/rec needs to be performed first."
            )
        artifact = state.get("papers")
    else:
        if not state.get("multi_papers"):
            logger.info("No multi papers found in state, raising NoPapersFoundError")
            raise NoPapersFoundError(
                "No papers found. A search/rec needs to be performed first."
            )
        artifact = state.get("multi_papers")
    content = "Papers displayed successfully."
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
