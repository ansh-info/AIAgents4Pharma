#!/usr/bin/env python3
"""
Tool for downloading bioRxiv paper metadata and downloading PDFs to temporary files,
with shared cloudscraper session and CF-challenge timeout.
"""

import logging
import tempfile
from typing import Annotated, Any, List

import hydra
import cloudscraper
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

    dois: List[str] = Field(description="List of DOIs for bioRxiv papers")
    tool_call_id: Annotated[str, InjectedToolCallId]


def _get_biorxiv_config() -> Any:
    """Load bioRxiv download configuration via Hydra."""
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["tools/download_biorxiv_paper=default"]
        )
    return cfg.tools.download_biorxiv_paper


def fetch_biorxiv_metadata(api_url: str, doi: str, timeout: int) -> dict:
    """Fetch metadata from bioRxiv 'details' API using cloudscraper too."""
    url = f"{api_url}/biorxiv/{doi}/na/json"
    logger.info("Fetching metadata for DOI %s from %s", doi, url)
    # We can reuse CF-bypass session here too if desired, but metadata endpoints rarely block
    resp = cloudscraper.create_scraper().get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def download_pdf_to_temp(
    cfg: Any,
    scraper: cloudscraper.CloudScraper,
    doi: str,
    version: str,
    timeout: int,
    chunk_size: int,
) -> tuple[str, str] | None:
    """
    Download PDF via a shared cloudscraper session.
    Returns (temp_path, filename) or None on failure.
    """
    landing = cfg.landing_url_template.format(doi=doi, version=version)
    pdf_url = cfg.pdf_url_template.format(doi=doi, version=version)

    logger.info("Downloading PDF for %s via %s", doi, pdf_url)
    try:
        # 1) Hit landing page so CF JS-challenge is solved
        r1 = scraper.get(landing, timeout=timeout)
        r1.raise_for_status()

        # 2) Stream the .full.pdf
        r2 = scraper.get(pdf_url, timeout=timeout, stream=True)
        r2.raise_for_status()

        # 3) Write into temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            for chunk in r2.iter_content(chunk_size=chunk_size):
                if chunk:
                    tmp.write(chunk)
            temp_path = tmp.name

        filename = doi.replace("/", "_").replace(".", "_") + ".pdf"
        logger.info("Saved PDF to %s", temp_path)
        return temp_path, filename

    except Exception as exc:
        logger.error("Failed to download %s: %s", doi, exc)
        return None


def extract_metadata(
    paper_data: dict, doi: str, pdf_result: tuple[str, str] | None
) -> dict:
    """Parse JSON and attach PDF download info."""
    coll = paper_data.get("collection", [])
    if not coll:
        raise RuntimeError(f"No data for DOI {doi}")
    paper = coll[0]

    # Basic fields
    title = paper.get("title", "N/A").strip()
    authors = [a.strip() for a in paper.get("authors", "").split(";") if a.strip()]
    abstract = paper.get("abstract", "N/A").strip()
    date = paper.get("date", "N/A").strip()
    category = paper.get("category", "N/A").strip()
    version = paper.get("version", "N/A")

    if pdf_result:
        temp_path, filename = pdf_result
        access = "open_access_downloaded"
    else:
        temp_path, filename = "", doi.replace("/", "_").replace(".", "_") + ".pdf"
        access = "download_failed"

    return {
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "Publication Date": date,
        "DOI": doi,
        "Category": category,
        "Version": version,
        "URL": temp_path,
        "pdf_url": temp_path,
        "filename": filename,
        "source": "biorxiv",
        "server": "biorxiv",
        "access_type": access,
        "temp_file_path": temp_path,
    }


def _get_snippet(text: str) -> str:
    if not text or text == "N/A":
        return ""
    sents = text.split(". ")
    snippet = ". ".join(sents[:2])
    if not snippet.endswith("."):
        snippet += "."
    return snippet


def _build_summary(data: dict[str, Any]) -> str:
    top = list(data.values())[:3]
    downloaded = sum(
        1 for x in data.values() if x["access_type"] == "open_access_downloaded"
    )
    lines = []
    for i, p in enumerate(top, start=1):
        snippet = _get_snippet(p["Abstract"])
        lines.append(
            f"{i}. {p['Title']} (DOI:{p['DOI']}, {p['Publication Date']})\n"
            f"   Category: {p['Category']}\n"
            f"   Access: {p['access_type']}\n"
            + (
                f"   Downloaded to: {p['temp_file_path']}\n"
                if p["temp_file_path"]
                else ""
            )
            + (f"   Abstract snippet: {snippet}" if snippet else "")
        )
    return (
        "Download completed. Metadata attached as an artifact.\n"
        f"Total papers: {len(data)}, PDFs downloaded: {downloaded}\n"
        "Top 3:\n" + "\n".join(lines)
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
    Get metadata + download PDFs for one or more bioRxiv DOIs.
    """
    logger.info("Starting download for DOIs: %s", dois)
    cfg = _get_biorxiv_config()
    api_url = cfg.api_url
    timeout = cfg.request_timeout
    chunk_size = cfg.chunk_size
    reuse_session = getattr(cfg, "session_reuse", True)
    cf_timeout = getattr(cfg, "cf_clearance_timeout", 10)

    # Build a shared scraper if desired
    scraper = None
    if reuse_session:
        scraper = cloudscraper.create_scraper(
            browser={"custom": cfg.user_agent}, delay=cf_timeout
        )

    results: dict[str, Any] = {}
    for doi in dois:
        try:
            # 1) metadata
            meta = fetch_biorxiv_metadata(api_url, doi, timeout)
            version = meta["collection"][0].get("version", "1")

            # 2) choose or build a scraper for PDF
            pdf_scraper = scraper or cloudscraper.create_scraper(
                browser={"custom": cfg.user_agent}, delay=cf_timeout
            )

            # 3) download
            pdf_res = download_pdf_to_temp(
                cfg, pdf_scraper, doi, version, timeout, chunk_size
            )

            # 4) record
            results[doi] = extract_metadata(meta, doi, pdf_res)

        except Exception as e:
            logger.warning("Error for %s: %s", doi, e)
            results[doi] = {
                "Title": "Error fetching",
                "Authors": [],
                "Abstract": str(e),
                "Publication Date": "N/A",
                "DOI": doi,
                "Category": "N/A",
                "Version": "N/A",
                "URL": "",
                "pdf_url": "",
                "filename": doi.replace("/", "_").replace(".", "_") + ".pdf",
                "source": "biorxiv",
                "server": "biorxiv",
                "access_type": "error",
                "temp_file_path": "",
                "error": str(e),
            }

    summary = _build_summary(results)
    return Command(
        update={
            "article_data": results,
            "messages": [
                ToolMessage(
                    content=summary,
                    tool_call_id=tool_call_id,
                    artifact=results,
                )
            ],
        }
    )
