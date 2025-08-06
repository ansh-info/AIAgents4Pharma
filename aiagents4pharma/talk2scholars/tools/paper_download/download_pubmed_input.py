#!/usr/bin/env python3
"""
Tool for downloading PubMed paper metadata and downloading PDFs to temporary files.
"""

import logging
import tempfile
import xml.etree.ElementTree as ET
from typing import Annotated, Any, List

import hydra
import requests
from bs4 import BeautifulSoup
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
    logger.info("Fetching metadata from ID converter for PMID %s: %s", pmid, query_url)
    response = requests.get(query_url, timeout=request_timeout)
    response.raise_for_status()
    result = response.json()
    logger.info("ID converter response for PMID %s: %s", pmid, result)
    return result


def try_alternative_pdf_sources(pmcid: str, request_timeout: int, cfg: Any) -> str:
    """Try alternative PDF sources when OA API fails."""

    # Strategy 1: Europe PMC Service
    europe_pmc_url = f"{cfg.europe_pmc_base_url}?accid={pmcid}&blobtype=pdf"
    logger.info("Trying Europe PMC service for %s: %s", pmcid, europe_pmc_url)
    try:
        response = requests.head(europe_pmc_url, timeout=request_timeout)
        if response.status_code == 200:
            logger.info("Europe PMC service works for %s", pmcid)
            return europe_pmc_url
    except Exception as e:
        logger.info("Europe PMC service failed for %s: %s", pmcid, str(e))

    # Strategy 2: Scrape PMC page for citation_pdf_url meta tag
    pmc_page_url = f"{cfg.pmc_page_base_url}/{pmcid}/"
    logger.info("Scraping PMC page for PDF meta tag for %s: %s", pmcid, pmc_page_url)
    try:
        headers = {"User-Agent": cfg.user_agent}
        response = requests.get(pmc_page_url, headers=headers, timeout=request_timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Look for citation_pdf_url meta tag
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta and pdf_meta.get("content"):
            pdf_url = pdf_meta.get("content")
            logger.info("Found citation_pdf_url meta tag for %s: %s", pmcid, pdf_url)
            return pdf_url

    except Exception as e:
        logger.info("PMC page scraping failed for %s: %s", pmcid, str(e))

    # Strategy 3: Direct PMC PDF URL pattern
    direct_pmc_url = f"{cfg.direct_pmc_pdf_base_url}/{pmcid}/pdf/"
    logger.info("Trying direct PMC PDF URL for %s: %s", pmcid, direct_pmc_url)
    try:
        response = requests.head(direct_pmc_url, timeout=request_timeout)
        if response.status_code == 200:
            logger.info("Direct PMC PDF URL works for %s", pmcid)
            return direct_pmc_url
    except Exception as e:
        logger.info("Direct PMC PDF URL failed for %s: %s", pmcid, str(e))

    logger.warning("All alternative PDF sources failed for %s", pmcid)
    return ""


def fetch_pdf_url(oa_api_url: str, pmcid: str, request_timeout: int, cfg: Any) -> str:
    """Fetch PDF URL from OA API with comprehensive fallback strategies."""
    if not pmcid or pmcid == "N/A":
        logger.info("No PMCID available for PDF fetch: %s", pmcid)
        return ""

    # Strategy 1: Official OA API (fastest when it works)
    query_url = f"{oa_api_url}?id={pmcid}"
    logger.info("Fetching PDF URL for PMCID %s from: %s", pmcid, query_url)
    try:
        response = requests.get(query_url, timeout=request_timeout)
        response.raise_for_status()

        logger.info(
            "OA API response for PMCID %s: %s", pmcid, response.text[:500]
        )  # Log first 500 chars

        # Parse XML response
        root = ET.fromstring(response.text)

        # Check for error first
        error_elem = root.find(".//error")
        if error_elem is not None:
            error_code = error_elem.get("code", "unknown")
            error_text = error_elem.text or "unknown error"
            logger.warning(
                "OA API error for PMCID %s: %s - %s. Trying alternatives...",
                pmcid,
                error_code,
                error_text,
            )
            return try_alternative_pdf_sources(pmcid, request_timeout, cfg)

        # Look for PDF link first (preferred)
        pdf_link = root.find(".//link[@format='pdf']")
        if pdf_link is not None:
            pdf_url = pdf_link.get("href", "")
            logger.info("Found PDF URL for PMCID %s: %s", pmcid, pdf_url)

            # Convert FTP links to HTTPS for download compatibility
            if pdf_url.startswith(cfg.ftp_base_url):
                pdf_url = pdf_url.replace(cfg.ftp_base_url, cfg.https_base_url)
                logger.info("Converted FTP to HTTPS for %s: %s", pmcid, pdf_url)

            return pdf_url

        # If no PDF link found, try alternatives
        logger.warning(
            "No PDF link found in OA API response for PMCID %s. Trying alternatives...",
            pmcid,
        )
        return try_alternative_pdf_sources(pmcid, request_timeout, cfg)

    except Exception as e:
        logger.warning(
            "OA API failed for %s: %s. Trying alternatives...", pmcid, str(e)
        )
        return try_alternative_pdf_sources(pmcid, request_timeout, cfg)


def download_pdf_to_temp(
    pdf_url: str, pmid: str, request_timeout: int, chunk_size: int = 8192
) -> tuple[str, str] | None:
    """
    Download PDF from URL to a temporary file.
    Returns tuple of (temp_file_path, filename) or None if failed.
    """
    if not pdf_url:
        logger.info("No PDF URL available for PMID %s", pmid)
        return None

    try:
        logger.info("Downloading PDF for PMID %s from %s", pmid, pdf_url)
        response = requests.get(pdf_url, timeout=request_timeout, stream=True)
        response.raise_for_status()

        # Download to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        logger.info("PubMed PDF downloaded to temporary file: %s", temp_file_path)

        # Determine filename from Content-Disposition header or default
        if "filename=" in response.headers.get("Content-Disposition", ""):
            filename = (
                response.headers.get("Content-Disposition", "")
                .split("filename=")[-1]
                .strip('"')
            )
        else:
            filename = f"pmid_{pmid}.pdf"

        return temp_file_path, filename

    except (requests.exceptions.RequestException, OSError) as e:
        logger.error("Failed to download PDF for PMID %s: %s", pmid, e)
        return None


def extract_metadata(
    id_data: dict, pmid: str, pdf_download_result: tuple[str, str] | None
) -> dict:
    """Extract metadata from the PMC ID Converter response and include download info."""
    if "records" not in id_data or not id_data["records"]:
        raise RuntimeError(f"No PMC data found for PMID {pmid}")

    record = id_data["records"][0]  # Get first (and should be only) record

    # Extract basic fields
    pmcid = record.get("pmcid", "N/A")
    doi = record.get("doi", "N/A")

    # Handle PDF download results
    if pdf_download_result:
        temp_file_path, filename = pdf_download_result
        access_type = "open_access_downloaded"
        pdf_url = temp_file_path  # Use local temp file path
    else:
        temp_file_path = ""
        filename = f"pmid_{pmid}.pdf"
        access_type = "abstract_only" if pmcid != "N/A" else "no_pmcid"
        pdf_url = ""

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
        "URL": pdf_url,  # Now points to local temp file or empty
        "pdf_url": pdf_url,  # Same as URL
        "access_type": access_type,
        "filename": filename,
        "source": "pubmed",
        "temp_file_path": temp_file_path,  # Explicit temp file path for cleanup if needed
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
    downloaded_count = sum(
        1
        for paper in article_data.values()
        if paper.get("access_type") == "open_access_downloaded"
    )

    for idx, paper in enumerate(top):
        title = paper.get("Title", "N/A")
        pmid = paper.get("PMID", "N/A")
        pmcid = paper.get("PMCID", "N/A")
        access_type = paper.get("access_type", "N/A")
        temp_file_path = paper.get("temp_file_path", "")
        snippet = _get_snippet(paper.get("Abstract", ""))

        line = f"{idx+1}. {title} (PMID: {pmid})"
        if pmcid != "N/A":
            line += f"\n   PMCID: {pmcid}"
        line += f"\n   Access: {access_type}"
        if temp_file_path:
            line += f"\n   Downloaded to: {temp_file_path}"
        if snippet:
            line += f"\n   Abstract snippet: {snippet}"
        lines.append(line)

    summary = "\n".join(lines)
    return (
        "Download was successful from PubMed. Papers metadata are attached as an artifact. "
        "Here is a summary of the results:\n"
        f"Number of papers found: {len(article_data)}\n"
        f"PDFs successfully downloaded: {downloaded_count}\n"
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
    Get metadata and download PDFs for one or more PubMed papers using their PMIDs.

    Args:
        pmids: List of PubMed ID strings (e.g., ['12345678', '87654321'])
    """
    logger.info(
        "Fetching metadata and downloading PDFs from PubMed for PMIDs: %s", pmids
    )

    # Load configuration
    cfg = _get_pubmed_config()
    id_converter_url = cfg.id_converter_url
    oa_api_url = cfg.oa_api_url
    request_timeout = cfg.request_timeout
    chunk_size = getattr(cfg, "chunk_size", 8192)  # Default chunk size

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

            pdf_url = (
                fetch_pdf_url(oa_api_url, pmcid, request_timeout, cfg) if pmcid else ""
            )

            # Step 3: Download PDF if URL is available
            pdf_download_result = None
            if pdf_url:
                pdf_download_result = download_pdf_to_temp(
                    pdf_url, pmid, request_timeout, chunk_size
                )

            # Step 4: Extract and structure metadata
            article_data[pmid] = extract_metadata(id_data, pmid, pdf_download_result)

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
