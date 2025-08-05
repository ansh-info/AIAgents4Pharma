#!/usr/bin/env python3
"""
Tool for downloading bioRxiv paper metadata and retrieving the PDF URL.
"""

import logging
from typing import Annotated, Any, List

import hydra
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadBiorxivPaperInput(BaseModel):
    """Input schema for the bioRxiv paper download tool."""

    dois: List[str] = Field(
        description="List of DOIs for bioRxiv papers (e.g., '10.1101/2020.09.09.20191205')"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Helper to load bioRxiv download configuration
def _get_biorxiv_config() -> Any:
    """Load bioRxiv download configuration."""
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_biorxiv_paper=default"]
        )
    return cfg.tools.download_biorxiv_paper


def fetch_biorxiv_metadata(api_url: str, doi: str, request_timeout: int) -> dict:
    """Fetch and parse metadata from the bioRxiv API."""
    # Note: bioRxiv API uses 'biorxiv' as the server parameter
    query_url = f"{api_url}/biorxiv/{doi}/na/json"
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def extract_metadata(paper_data: dict, doi: str) -> dict:
    """Extract metadata from the JSON response."""
    # The API returns a collection with papers in a 'collection' key
    if "collection" not in paper_data or not paper_data["collection"]:
        raise RuntimeError(f"No paper data found for DOI {doi}")

    paper = paper_data["collection"][0]  # Get first (and should be only) paper

    title = paper.get("title", "N/A").strip()

    # Authors are typically in a semicolon-separated string
    authors_str = paper.get("authors", "")
    authors = (
        [author.strip() for author in authors_str.split(";") if author.strip()]
        if authors_str
        else []
    )

    abstract = paper.get("abstract", "N/A").strip()
    pub_date = paper.get("date", "N/A").strip()
    category = paper.get("category", "N/A").strip()
    version = paper.get("version", "N/A")

    # Construct bioRxiv PDF URL
    # Format: https://www.biorxiv.org/content/{DOI}v{version}.full.pdf
    pdf_url = f"https://www.biorxiv.org/content/{doi}v{version}.full.pdf"

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "DOI": doi,
        "Category": category,
        "Version": version,
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "filename": f"{doi.replace('/', '_').replace('.', '_')}.pdf",
        "source": "biorxiv",
        "server": "biorxiv",
    }


def _get_snippet(abstract: str) -> str:
    """Extract the first one or two sentences from an abstract."""
    if not abstract or abstract == "N/A":
        return ""
    sentences = abstract.split(". ")
    snippet_sentences = sentences[:2]
    snippet = ". ".join(snippet_sentences)
    if not snippet.endswith("."):
        snippet += "."
    return snippet


def _build_summary(article_data: dict[str, Any]) -> str:
    """Build a summary string for up to three papers with snippets."""
    top = list(article_data.values())[:3]
    lines: list[str] = []
    for idx, paper in enumerate(top):
        title = paper.get("Title", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        url = paper.get("URL", "")
        category = paper.get("Category", "N/A")
        snippet = _get_snippet(paper.get("Abstract", ""))
        line = f"{idx+1}. {title} ({pub_date})"
        if category != "N/A":
            line += f"\n   Category: {category}"
        if url:
            line += f"\n   View PDF: {url}"
        if snippet:
            line += f"\n   Abstract snippet: {snippet}"
        lines.append(line)
    summary = "\n".join(lines)
    return (
        "Download was successful from bioRxiv. Papers metadata are attached as an artifact. "
        "Here is a summary of the results:\n"
        f"Number of papers found: {len(article_data)}\n"
        "Top 3 papers:\n" + summary
    )


@tool(
    args_schema=DownloadBiorxivPaperInput,
    parse_docstring=True,
)
def download_biorxiv_paper(
    dois: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URLs for one or more bioRxiv papers using their DOIs.

    Args:
        dois: List of DOI strings (e.g., ['10.1101/2020.09.09.20191205'])
    """
    logger.info("Fetching metadata from bioRxiv for DOIs: %s", dois)

    # Load configuration
    cfg = _get_biorxiv_config()
    api_url = cfg.api_url
    request_timeout = cfg.request_timeout

    # Aggregate results
    article_data: dict[str, Any] = {}
    for doi in dois:
        logger.info("Processing DOI: %s from bioRxiv", doi)
        try:
            # Fetch and parse metadata
            paper_data = fetch_biorxiv_metadata(api_url, doi, request_timeout)
            article_data[doi] = extract_metadata(paper_data, doi)
        except Exception as e:
            logger.warning("Error processing DOI %s: %s", doi, str(e))
            # Add placeholder data for failed DOIs
            article_data[doi] = {
                "Title": "Error fetching paper",
                "Authors": [],
                "Abstract": f"Error: {str(e)}",
                "Publication Date": "N/A",
                "DOI": doi,
                "Category": "N/A",
                "Version": "N/A",
                "URL": "",
                "pdf_url": "",
                "filename": f"{doi.replace('/', '_').replace('.', '_')}.pdf",
                "source": "biorxiv",
                "server": "biorxiv",
                "error": str(e),
            }

    # Build and return summary
    content = _build_summary(article_data)
    return Command(
        update={
            "article_data": article_data,
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=article_data,
                )
            ],
        }
    )
