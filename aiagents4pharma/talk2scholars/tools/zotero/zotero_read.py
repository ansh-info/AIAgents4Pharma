#!/usr/bin/env python3

"""
This tool is used to search for papers in Zotero library.
"""

import logging
from typing import Annotated, Any, Dict
import hydra
from pyzotero import zotero
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZoteroSearchInput(BaseModel):
    """Input schema for the Zotero search tool."""

    query: str = Field(
        description="Search query string to find papers in Zotero library."
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Load hydra configuration
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(config_name="config", overrides=["tools/zotero=default"])
    cfg = cfg.tools.zotero


@tool(args_schema=ZoteroSearchInput)
def zotero_search_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """
    Search for papers in Zotero library.

    Args:
        query (str): The search query string to find papers.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of results to return. Defaults to 2.

    Returns:
        Dict[str, Any]: The search results and related information.
    """
    logger.info("Starting Zotero paper search...")

    # Initialize Zotero client
    zot = zotero.Zotero(cfg.user_id, cfg.library_type, cfg.api_key)

    # Get items matching the query
    items = zot.items(q=query, limit=min(limit, 100))
    logger.info("Received %d items from Zotero", len(items))

    # Filter and format papers
    filtered_papers = {}
    for item in items:
        data = item.get("data", {})
        if data.get("itemType") == "journalArticle":
            item_key = data.get("key")
            if item_key:
                filtered_papers[item_key] = {
                    "title": data.get("title", "N/A"),
                    "abstract": data.get("abstractNote", "N/A"),
                    "date": data.get("date", "N/A"),
                    "url": data.get("url", "N/A"),
                    "type": data.get("itemType", "N/A"),
                }

    logger.info("Filtered %d journal articles", len(filtered_papers))

    # # Filter and format papers
    # filtered_papers = {}
    #
    # # Process each item
    # for item in items:
    #     if not isinstance(item, dict):
    #         continue
    #
    #     data = item.get("data", {})
    #     if not data or data.get("itemType") != cfg.item_type:
    #         continue
    #
    #     key = data.get("key")
    #     if not key:
    #         continue
    #
    #     filtered_papers[key] = {
    #         "title": data.get("title", "N/A"),
    #         "abstract": data.get("abstractNote", "N/A"),
    #         "date": data.get("date", "N/A"),
    #         "url": data.get("url", "N/A"),
    #         "type": data.get("itemType", "N/A"),
    #     }
    #
    # logger.info("Filtered %d papers from Zotero", len(filtered_papers))

    return Command(
        update={
            "zotero_papers": filtered_papers,
            "messages": [
                ToolMessage(
                    content=f"Search Successful: {filtered_papers}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
