#!/usr/bin/env python3
"""
Tool for downloading arXiv paper metadata and downloading PDFs to temporary files.
"""

import logging
import tempfile
import xml.etree.ElementTree as ET
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


class DownloadArxivPaperInput(BaseModel):
    """Input schema for the arXiv paper download tool."""

    arxiv_ids: List[str] = Field(
        description="List of arXiv paper IDs used to retrieve paper details and PDF URLs."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Helper to load arXiv download configuration
def _get_arxiv_config() -> Any:
    """Load arXiv download configuration."""
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_arxiv_paper=default"]
        )
    return cfg.tools.download_arxiv_paper


def fetch_arxiv_metadata(
    api_url: str, arxiv_id: str, request_timeout: int
) -> ET.Element:
    """Fetch and parse metadata from the arXiv API."""
    query_url = f"{api_url}?search_query=id:{arxiv_id}&start=0&max_results=1"
    logger.info("Fetching metadata for arXiv ID %s from: %s", arxiv_id, query_url)
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return ET.fromstring(response.text)


def download_pdf_to_temp(
    pdf_url: str, arxiv_id: str, request_timeout: int, cfg: Any, chunk_size: int = 8192
) -> tuple[str, str] | None:
    """
    Download PDF from URL to a temporary file.
    Returns tuple of (temp_file_path, filename) or None if failed.
    """
    if not pdf_url:
        logger.info("No PDF URL available for arXiv ID %s", arxiv_id)
        return None

    try:
        logger.info("Downloading PDF for arXiv ID %s from %s", arxiv_id, pdf_url)

        # Use proper headers for better compatibility
        headers = {"User-Agent": cfg.user_agent}
        response = requests.get(
            pdf_url, headers=headers, timeout=request_timeout, stream=True
        )
        response.raise_for_status()

        # Download to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    temp_file.write(chunk)
            temp_file_path = temp_file.name

        logger.info("arXiv PDF downloaded to temporary file: %s", temp_file_path)

        # Determine filename - arXiv rarely provides Content-Disposition headers
        filename = f"{arxiv_id}.pdf"  # Use arXiv ID as filename

        return temp_file_path, filename

    except (requests.exceptions.RequestException, OSError) as e:
        logger.error("Failed to download PDF for arXiv ID %s: %s", arxiv_id, e)
        return None


def extract_metadata(
    entry: ET.Element,
    ns: dict,
    arxiv_id: str,
    pdf_download_result: tuple[str, str] | None,
) -> dict:
    """Extract metadata from the XML entry and include download info."""
    title_elem = entry.find("atom:title", ns)
    title = (title_elem.text or "").strip() if title_elem is not None else "N/A"

    authors = []
    for author_elem in entry.findall("atom:author", ns):
        name_elem = author_elem.find("atom:name", ns)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())

    summary_elem = entry.find("atom:summary", ns)
    abstract = (summary_elem.text or "").strip() if summary_elem is not None else "N/A"

    published_elem = entry.find("atom:published", ns)
    pub_date = (
        (published_elem.text or "").strip() if published_elem is not None else "N/A"
    )

    # Handle PDF download results
    if pdf_download_result:
        temp_file_path, filename = pdf_download_result
        pdf_url = temp_file_path  # Use local temp file path
        access_type = "open_access_downloaded"
    else:
        temp_file_path = ""
        filename = f"{arxiv_id}.pdf"
        pdf_url = ""
        access_type = "download_failed"

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "URL": pdf_url,  # Now points to local temp file or empty
        "pdf_url": pdf_url,  # Same as URL
        "filename": filename,
        "source": "arxiv",
        "arxiv_id": arxiv_id,
        "access_type": access_type,
        "temp_file_path": temp_file_path,  # Explicit temp file path for cleanup if needed
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
    downloaded_count = sum(
        1
        for paper in article_data.values()
        if paper.get("access_type") == "open_access_downloaded"
    )

    for idx, paper in enumerate(top):
        title = paper.get("Title", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        arxiv_id = paper.get("arxiv_id", "N/A")
        access_type = paper.get("access_type", "N/A")
        temp_file_path = paper.get("temp_file_path", "")
        snippet = _get_snippet(paper.get("Abstract", ""))

        line = f"{idx+1}. {title} (arXiv:{arxiv_id}, {pub_date})"
        line += f"\n   Access: {access_type}"
        if temp_file_path:
            line += f"\n   Downloaded to: {temp_file_path}"
        if snippet:
            line += f"\n   Abstract snippet: {snippet}"
        lines.append(line)

    summary = "\n".join(lines)
    return (
        "Download was successful from arXiv. Papers metadata are attached as an artifact. "
        "Here is a summary of the results:\n"
        f"Number of papers found: {len(article_data)}\n"
        f"PDFs successfully downloaded: {downloaded_count}\n"
        "Top 3 papers:\n" + summary
    )


@tool(
    args_schema=DownloadArxivPaperInput,
    parse_docstring=True,
)
def download_arxiv_paper(
    arxiv_ids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and download PDFs for one or more arXiv papers using their unique arXiv IDs.
    """
    logger.info(
        "Fetching metadata and downloading PDFs from arXiv for paper IDs: %s", arxiv_ids
    )

    # Load configuration
    cfg = _get_arxiv_config()
    api_url = cfg.api_url
    request_timeout = cfg.request_timeout
    chunk_size = getattr(cfg, "chunk_size", 8192)  # Default chunk size
    pdf_base_url = cfg.pdf_base_url

    # Aggregate results
    article_data: dict[str, Any] = {}
    for aid in arxiv_ids:
        logger.info("Processing arXiv ID: %s", aid)
        try:
            # Step 1: Fetch and parse metadata
            root = fetch_arxiv_metadata(api_url, aid, request_timeout)
            entry = root.find("atom:entry", {"atom": "http://www.w3.org/2005/Atom"})

            if entry is None:
                logger.warning("No entry found for arXiv ID %s", aid)
                # Add error entry
                article_data[aid] = {
                    "Title": "Error fetching paper",
                    "Authors": [],
                    "Abstract": "No entry found in arXiv API response",
                    "Publication Date": "N/A",
                    "URL": "",
                    "pdf_url": "",
                    "filename": f"{aid}.pdf",
                    "source": "arxiv",
                    "arxiv_id": aid,
                    "access_type": "error",
                    "temp_file_path": "",
                    "error": "No entry found in arXiv API response",
                }
                continue

            # Step 2: Extract PDF URL from metadata
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            pdf_url = next(
                (
                    link.attrib.get("href")
                    for link in entry.findall("atom:link", ns)
                    if link.attrib.get("title") == "pdf"
                ),
                None,
            )

            # Step 3: Fallback to constructed PDF URL if not found in metadata
            if not pdf_url:
                pdf_url = f"{pdf_base_url}/{aid}.pdf"
                logger.info("Using constructed PDF URL for %s: %s", aid, pdf_url)

            # Step 4: Download PDF if URL is available
            pdf_download_result = None
            if pdf_url:
                pdf_download_result = download_pdf_to_temp(
                    pdf_url, aid, request_timeout, cfg, chunk_size
                )

            # Step 5: Extract and structure metadata
            article_data[aid] = extract_metadata(entry, ns, aid, pdf_download_result)

        except Exception as e:
            logger.warning("Error processing arXiv ID %s: %s", aid, str(e))
            # Add placeholder data for failed arXiv IDs
            article_data[aid] = {
                "Title": "Error fetching paper",
                "Authors": [],
                "Abstract": f"Error: {str(e)}",
                "Publication Date": "N/A",
                "URL": "",
                "pdf_url": "",
                "filename": f"{aid}.pdf",
                "source": "arxiv",
                "arxiv_id": aid,
                "access_type": "error",
                "temp_file_path": "",
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
