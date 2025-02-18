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
    only_articles: bool = Field(
        default=True, description="Whether to only search for journal articles/"
        "conference papers."
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Load hydra configuration
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(config_name="config", overrides=["tools/zotero=default"])
    cfg = cfg.tools.zotero

@tool(args_schema=ZoteroSearchInput, parse_docstring=True)
def zotero_search_tool(
    query: str,
    only_articles: bool,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
) -> Dict[str, Any]:
    """
    Use this tool to search and retrieve papers from Zotero library.

    Args:
        query (str): The search query string to find papers.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of results to return. Defaults to 2.

    Returns:
        Dict[str, Any]: The search results and related information.
    """
    logger.info("Starting Zotero search with query and filter: %s, %s",
                query,
                only_articles)

    # Initialize Zotero client
    zot = zotero.Zotero(cfg.user_id, cfg.library_type, cfg.api_key)

    # Get items matching the query
    items = zot.items(q=query, limit=min(limit, 100))
    logger.info("Received %d items from Zotero", len(items))

    # Filter only articles
    filter_item_types = None
    if only_articles:
        filter_item_types = ["journalArticle", "conferencePaper", "preprint"]

    # Filter and format papers
    filtered_papers = {}
    for item in items:
        data = item.get("data", {})
        # Filter only articles
        if only_articles:
            if data.get("itemType") not in filter_item_types:
                continue
        # Add to filtered papers
        item_key = data.get("key")
        if item_key:
            filtered_papers[item_key] = {
                "Title": data.get("title", "N/A"),
                "Abstract": data.get("abstractNote", "N/A"),
                "Date": data.get("date", "N/A"),
                "URL": data.get("url", "N/A"),
                "Type": data.get("itemType", "N/A"),
            }

    logger.info("Filtered %d items", len(filtered_papers))

    content = "Retrieval was successful. Papers are attached as an artifact."
    content += " And here is a summary of the retrieval results:"
    content += f"Number of papers found: {len(filtered_papers)}\n"
    content += f"Query: {query}\n"

    return Command(
        update={
            "zotero_papers": filtered_papers,
            "last_displayed_papers": "zotero_papers",
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
