#!/usr/bin/env python3

"""
This tool is used to display the table of studies.
"""

import logging
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""

@tool("last_displayed_papers", parse_docstring=True)
def last_displayed_papers(state: Annotated[dict, InjectedState]) -> str:
    """
    Access the last displayed papers in the state.
    If no papers are found, raises an exception

    Use this also to get the last displayed papers from the state,
    and then use the papers to get recommendations for a single paper or
    multiple papers.

    Args:
        state (dict): The state of the agent containing the papers.

    Returns:
        str: A message with the last displayed papers.

    Raises:
        NoPapersFoundError: If no papers are found in the state.
    """
    logger.info("Accessing last displayed papers from the state")

    if not state.get("last_displayed_papers"):
        logger.info("No papers found in state, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search needs to be performed first."
        )
    return state.get("last_displayed_papers")
