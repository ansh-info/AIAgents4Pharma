#!/usr/bin/env python3
"""
Tool for downloading PubMed paper metadata and retrieving the PDF URL.
"""

import logging
from typing import Annotated, Any, List
import xml.etree.ElementTree as ET

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


class DownloadPubmedPaperInput(BaseModel):
    """Input schema for the PubMed paper download tool."""

    pmids: List[str] = Field(
        description="List of PubMed IDs (PMIDs) (e.g., ['12345678', '87654321'])"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Helper to load PubMed download configuration
def _get_pubmed_config() -> Any:
    """Load PubMed download configuration."""
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_pubmed_paper=default"]
        )
    return cfg.tools.download_pubmed_paper


def fetch_pmcid_metadata(
    id_converter_url: str, pmid: str, request_timeout: int
) -> dict:
    """Fetch metadata and PMCID from PMC ID Converter API."""
    query_url = f"{id_converter_url}?ids={pmid}&format=json"
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def fetch_pdf_url(oa_api_url: str, pmcid: str, request_timeout: int) -> str:
    """Fetch PDF URL from OA API if available, with FTP-to-HTTPS conversion."""
    if not pmcid or pmcid == "N/A":
        return ""

    query_url = f"{oa_api_url}?id={pmcid}"  # removed ?format=pdf which is invalid
    try:
        response = requests.get(query_url, timeout=request_timeout)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)

        # Look for PDF link
        pdf_link = root.find(".//link[@format='pdf']")
        if pdf_link is not None:
            pdf_url = pdf_link.get("href", "")

            # Convert FTP links to HTTPS
            if pdf_url.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
                pdf_url = pdf_url.replace(
                    "ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov"
                )
                logger.info("Converted FTP to HTTPS for %s: %s", pmcid, pdf_url)

            return pdf_url

        return ""
    except Exception as e:
        logger.warning("Could not fetch PDF URL for %s: %s", pmcid, str(e))
        return ""


def extract_metadata(id_data: dict, pmid: str, pdf_url: str) -> dict:
    """Extract metadata from the PMC ID Converter response."""
    if "records" not in id_data or not id_data["records"]:
        raise RuntimeError(f"No PMC data found for PMID {pmid}")

    record = id_data["records"][0]  # Get first (and should be only) record

    # Extract basic fields
    pmcid = record.get("pmcid", "N/A")
    doi = record.get("doi", "N/A")

    # Determine access type
    access_type = "open_access" if pmcid != "N/A" and pdf_url else "abstract_only"

    # For PubMed, we don't get title/authors from ID converter
    # In a real implementation, you might want to call E-utilities for full metadata
    # For now, we'll use placeholders and focus on the ID conversion functionality

    return {
        "Title": f"PubMed Article {pmid}",  # Placeholder - would need E-utilities for real title
        "Authors": [],  # Placeholder - would need E-utilities for real authors
        "Abstract": "Abstract available in PubMed",  # Placeholder
        "Publication Date": "N/A",  # Would need E-utilities for this
        "PMID": pmid,
        "PMCID": pmcid,
        "DOI": doi,
        "Journal": "N/A",  # Would need E-utilities for this
        "URL": pdf_url,
        "pdf_url": pdf_url,
        "access_type": access_type,
        "filename": f"pmid_{pmid}.pdf",
        "source": "pubmed",
    }


def _get_snippet(abstract: str) -> str:
    """Extract the first one or two sentences from an abstract."""
    if not abstract or abstract == "N/A" or abstract == "Abstract available in PubMed":
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
        pmid = paper.get("PMID", "N/A")
        pmcid = paper.get("PMCID", "N/A")
        access_type = paper.get("access_type", "N/A")
        url = paper.get("URL", "")
        snippet = _get_snippet(paper.get("Abstract", ""))

        line = f"{idx+1}. {title} (PMID: {pmid})"
        if pmcid != "N/A":
            line += f"\n   PMCID: {pmcid}"
        line += f"\n   Access: {access_type}"
        if url:
            line += f"\n   View PDF: {url}"
        if snippet:
            line += f"\n   Abstract snippet: {snippet}"
        lines.append(line)

    summary = "\n".join(lines)
    return (
        "Download was successful from PubMed. Papers metadata are attached as an artifact. "
        "Here is a summary of the results:\n"
        f"Number of papers found: {len(article_data)}\n"
        "Top 3 papers:\n" + summary
    )


@tool(
    args_schema=DownloadPubmedPaperInput,
    parse_docstring=True,
)
def download_pubmed_paper(
    pmids: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Get metadata and PDF URLs for one or more PubMed papers using their PMIDs.

    Args:
        pmids: List of PubMed ID strings (e.g., ['12345678', '87654321'])
    """
    logger.info("Fetching metadata from PubMed for PMIDs: %s", pmids)

    # Load configuration
    cfg = _get_pubmed_config()
    id_converter_url = cfg.id_converter_url
    oa_api_url = cfg.oa_api_url
    request_timeout = cfg.request_timeout

    # Aggregate results
    article_data: dict[str, Any] = {}
    for pmid in pmids:
        logger.info("Processing PMID: %s", pmid)
        try:
            # Step 1: Get PMCID and basic metadata
            id_data = fetch_pmcid_metadata(id_converter_url, pmid, request_timeout)

            # Step 2: Get PDF URL if PMCID exists
            pmcid = ""
            if "records" in id_data and id_data["records"]:
                pmcid = id_data["records"][0].get("pmcid", "")

            pdf_url = fetch_pdf_url(oa_api_url, pmcid, request_timeout) if pmcid else ""

            # Step 3: Extract and structure metadata
            article_data[pmid] = extract_metadata(id_data, pmid, pdf_url)

        except Exception as e:
            logger.warning("Error processing PMID %s: %s", pmid, str(e))
            # Add placeholder data for failed PMIDs
            article_data[pmid] = {
                "Title": "Error fetching paper",
                "Authors": [],
                "Abstract": f"Error: {str(e)}",
                "Publication Date": "N/A",
                "PMID": pmid,
                "PMCID": "N/A",
                "DOI": "N/A",
                "Journal": "N/A",
                "URL": "",
                "pdf_url": "",
                "access_type": "error",
                "filename": f"pmid_{pmid}.pdf",
                "source": "pubmed",
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
