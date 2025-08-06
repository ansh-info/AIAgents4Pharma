#!/usr/bin/env python3
"""
Tool for downloading medRxiv paper metadata and downloading PDFs to temporary files.
"""

import logging
import tempfile
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


class DownloadMedrxivPaperInput(BaseModel):
    """Input schema for the medRxiv paper download tool."""

    dois: List[str] = Field(
        description="List of DOIs for medRxiv papers (e.g., '10.1101/2020.09.09.20191205')"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Helper to load medRxiv download configuration
def _get_medrxiv_config() -> Any:
    """Load medRxiv download configuration."""
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_medrxiv_paper=default"]
        )
    return cfg.tools.download_medrxiv_paper


def fetch_medrxiv_metadata(api_url: str, doi: str, request_timeout: int) -> dict:
    """Fetch and parse metadata from the medRxiv API."""
    query_url = f"{api_url}/medrxiv/{doi}/na/json"
    logger.info("Fetching metadata for DOI %s from: %s", doi, query_url)
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def download_pdf_to_temp(
    pdf_url: str, doi: str, request_timeout: int, cfg: Any, chunk_size: int = 8192
) -> tuple[str, str] | None:
    """
    Download PDF from URL to a temporary file.
    Returns tuple of (temp_file_path, filename) or None if failed.
    """
    if not pdf_url:
        logger.info("No PDF URL available for DOI %s", doi)
        return None

    try:
        logger.info("Downloading PDF for DOI %s from %s", doi, pdf_url)

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

        logger.info("medRxiv PDF downloaded to temporary file: %s", temp_file_path)

        # Determine filename from Content-Disposition header or default
        filename = f"{doi.replace('/', '_').replace('.', '_')}.pdf"  # Default fallback

        content_disposition = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disposition:
            # Extract filename from Content-Disposition header
            try:
                import re

                filename_match = re.search(
                    r'filename[*]?=(?:"([^"]+)"|([^;]+))', content_disposition
                )
                if filename_match:
                    extracted_filename = filename_match.group(
                        1
                    ) or filename_match.group(2)
                    extracted_filename = extracted_filename.strip().strip('"')
                    if extracted_filename and extracted_filename.endswith(".pdf"):
                        filename = extracted_filename
                        logger.info("Extracted filename from header: %s", filename)
            except Exception as e:
                logger.warning(
                    "Failed to extract filename from Content-Disposition: %s", e
                )

        return temp_file_path, filename

    except (requests.exceptions.RequestException, OSError) as e:
        logger.error("Failed to download PDF for DOI %s: %s", doi, e)
        return None


def extract_metadata(
    paper_data: dict, doi: str, pdf_download_result: tuple[str, str] | None
) -> dict:
    """Extract metadata from the JSON response and include download info."""
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

    # Handle PDF download results
    if pdf_download_result:
        temp_file_path, filename = pdf_download_result
        pdf_url = temp_file_path  # Use local temp file path
        access_type = "open_access_downloaded"
    else:
        temp_file_path = ""
        filename = f"{doi.replace('/', '_').replace('.', '_')}.pdf"
        pdf_url = ""
        access_type = "download_failed"

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": pub_date,
        "DOI": doi,
        "Category": category,
        "Version": version,
        "URL": pdf_url,  # Now points to local temp file or empty
        "pdf_url": pdf_url,  # Same as URL
        "filename": filename,
        "source": "medrxiv",
        "server": "medrxiv",
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
        doi = paper.get("DOI", "N/A")
        category = paper.get("Category", "N/A")
        access_type = paper.get("access_type", "N/A")
        temp_file_path = paper.get("temp_file_path", "")
        snippet = _get_snippet(paper.get("Abstract", ""))

        line = f"{idx+1}. {title} (DOI:{doi}, {pub_date})"
        if category != "N/A":
            line += f"\n   Category: {category}"
        line += f"\n   Access: {access_type}"
        if temp_file_path:
            line += f"\n   Downloaded to: {temp_file_path}"
        if snippet:
            line += f"\n   Abstract snippet: {snippet}"
        lines.append(line)

    summary = "\n".join(lines)
    return (
        "Download was successful from medRxiv. Papers metadata are attached as an artifact. "
        "Here is a summary of the results:\n"
        f"Number of papers found: {len(article_data)}\n"
        f"PDFs successfully downloaded: {downloaded_count}\n"
        "Top 3 papers:\n" + summary
    )


@tool(
    args_schema=DownloadMedrxivPaperInput,
    parse_docstring=True,
)
def download_medrxiv_paper(
    dois: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and download PDFs for one or more medRxiv papers using their DOIs.

    Args:
        dois: List of DOI strings (e.g., ['10.1101/2020.09.09.20191205'])
    """
    logger.info(
        "Fetching metadata and downloading PDFs from medRxiv for DOIs: %s", dois
    )

    # Load configuration
    cfg = _get_medrxiv_config()
    api_url = cfg.api_url
    request_timeout = cfg.request_timeout
    chunk_size = getattr(cfg, "chunk_size", 8192)  # Default chunk size

    # Aggregate results
    article_data: dict[str, Any] = {}
    for doi in dois:
        logger.info("Processing DOI: %s from medRxiv", doi)
        try:
            # Step 1: Fetch and parse metadata
            paper_data = fetch_medrxiv_metadata(api_url, doi, request_timeout)

            if "collection" not in paper_data or not paper_data["collection"]:
                logger.warning("No paper data found for DOI %s", doi)
                # Add error entry
                article_data[doi] = {
                    "Title": "Error fetching paper",
                    "Authors": [],
                    "Abstract": "No paper data found in medRxiv API response",
                    "Publication Date": "N/A",
                    "DOI": doi,
                    "Category": "N/A",
                    "Version": "N/A",
                    "URL": "",
                    "pdf_url": "",
                    "filename": f"{doi.replace('/', '_').replace('.', '_')}.pdf",
                    "source": "medrxiv",
                    "server": "medrxiv",
                    "access_type": "error",
                    "temp_file_path": "",
                    "error": "No paper data found in medRxiv API response",
                }
                continue

            # Step 2: Extract version from metadata for PDF URL construction
            paper = paper_data["collection"][0]
            version = paper.get("version", "1")  # Default to version 1

            # Step 3: Construct medRxiv PDF URL
            # Format: https://www.medrxiv.org/content/{DOI}v{version}.full.pdf
            pdf_url = f"https://www.medrxiv.org/content/{doi}v{version}.full.pdf"

            # Step 4: Download PDF if URL is available
            pdf_download_result = None
            if pdf_url:
                pdf_download_result = download_pdf_to_temp(
                    pdf_url, doi, request_timeout, cfg, chunk_size
                )

            # Step 5: Extract and structure metadata
            article_data[doi] = extract_metadata(paper_data, doi, pdf_download_result)

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
                "source": "medrxiv",
                "server": "medrxiv",
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
