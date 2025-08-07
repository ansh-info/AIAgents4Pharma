#!/usr/bin/env python3
"""
Unified paper download tool for LangGraph.
Supports downloading papers from arXiv, medRxiv, and PubMed through a single interface.
"""

import logging
from typing import Annotated, Any, List, Literal

import hydra
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

from .arxiv_downloader import ArxivDownloader
from .base_paper_downloader import BasePaperDownloader
from .medrxiv_downloader import MedrxivDownloader
from .pubmed_downloader import PubmedDownloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedPaperDownloadInput(BaseModel):
    """Input schema for the unified paper download tool."""

    service: Literal["arxiv", "medrxiv", "pubmed"] = Field(
        description="Paper service to download from: 'arxiv', 'medrxiv', or 'pubmed'"
    )
    identifiers: List[str] = Field(
        description=(
            "List of paper identifiers. Format depends on service:\n"
            "- arxiv: arXiv IDs (e.g., ['1234.5678', '2301.12345'])\n"
            "- medrxiv: DOIs (e.g., ['10.1101/2020.09.09.20191205'])\n"
            "- pubmed: PMIDs (e.g., ['12345678', '87654321'])"
        )
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


class PaperDownloaderFactory:
    """Factory class for creating paper downloader instances."""

    @staticmethod
    def create(service: str) -> BasePaperDownloader:
        """
        Create appropriate downloader instance for the specified service.

        Args:
            service: Service name ('arxiv', 'medrxiv', 'pubmed')

        Returns:
            Configured downloader instance

        Raises:
            ValueError: If service is not supported
        """
        config = PaperDownloaderFactory._get_unified_config()
        service_config = PaperDownloaderFactory._build_service_config(config, service)

        if service == "arxiv":
            return ArxivDownloader(service_config)
        elif service == "medrxiv":
            return MedrxivDownloader(service_config)
        elif service == "pubmed":
            return PubmedDownloader(service_config)
        else:
            supported = config.tool.supported_services
            raise ValueError(f"Unsupported service: {service}. Supported: {supported}")

    @staticmethod
    def _get_unified_config() -> Any:
        """
        Load unified paper download configuration using Hydra.

        Returns:
            Unified configuration object
        """
        try:
            with hydra.initialize(version_base=None, config_path="../../configs"):
                cfg = hydra.compose(
                    config_name="config", overrides=["tools/paper_download=default"]
                )
            return cfg.tools.paper_download

        except Exception as e:
            logger.error("Failed to load unified paper download configuration: %s", e)
            raise RuntimeError(f"Configuration loading failed: {e}")

    @staticmethod
    def _build_service_config(unified_config: Any, service: str) -> Any:
        """
        Build service-specific configuration by merging common and service settings.

        Args:
            unified_config: The unified configuration object
            service: Service name

        Returns:
            Service-specific configuration object
        """
        if service not in unified_config.services:
            raise ValueError(f"Service '{service}' not found in configuration")

        # Create a simple config object that combines common and service-specific settings
        class ServiceConfig:
            def __init__(self, common_config, service_config):
                # Add all common settings
                for key, value in common_config.items():
                    setattr(self, key, value)

                # Add all service-specific settings (these override common if there's overlap)
                for key, value in service_config.items():
                    setattr(self, key, value)

        return ServiceConfig(unified_config.common, unified_config.services[service])


@tool(
    args_schema=UnifiedPaperDownloadInput,
    parse_docstring=True,
)
def download_papers(
    service: Literal["arxiv", "medrxiv", "pubmed"],
    identifiers: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Universal paper download tool supporting multiple academic paper services.

    Downloads paper metadata and PDFs from arXiv, medRxiv, or PubMed and stores them
    in temporary files for further processing. The downloaded PDFs can be accessed
    using the temp_file_path in the returned metadata.

    Args:
        service: Paper service to download from
            - 'arxiv': For arXiv preprints (requires arXiv IDs)
            - 'medrxiv': For medRxiv preprints (requires DOIs)
            - 'pubmed': For PubMed papers (requires PMIDs)
        identifiers: List of paper identifiers in the format expected by the service

    Returns:
        Command with article_data containing paper metadata and local file paths

    Examples:
        # Download from arXiv
        download_papers("arxiv", ["1234.5678", "2301.12345"])

        # Download from medRxiv
        download_papers("medrxiv", ["10.1101/2020.09.09.20191205"])

        # Download from PubMed
        download_papers("pubmed", ["12345678", "87654321"])
    """
    return _download_papers_impl(service, identifiers, tool_call_id)


# Convenience functions for backward compatibility (optional)
def download_arxiv_papers(
    arxiv_ids: List[str], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Convenience function for downloading arXiv papers."""
    return _download_papers_impl("arxiv", arxiv_ids, tool_call_id)


def download_medrxiv_papers(
    dois: List[str], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Convenience function for downloading medRxiv papers."""
    return _download_papers_impl("medrxiv", dois, tool_call_id)


def download_pubmed_papers(
    pmids: List[str], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Convenience function for downloading PubMed papers."""
    return _download_papers_impl("pubmed", pmids, tool_call_id)


def _download_papers_impl(
    service: Literal["arxiv", "medrxiv", "pubmed"],
    identifiers: List[str],
    tool_call_id: str,
) -> Command[Any]:
    """
    Internal implementation function that contains the actual download logic.
    This is called by both the decorated tool and the convenience functions.
    """
    logger.info(
        "Starting unified paper download for service '%s' with %d identifiers: %s",
        service,
        len(identifiers),
        identifiers,
    )

    try:
        # Step 1: Create appropriate downloader using factory
        downloader = PaperDownloaderFactory.create(service)
        logger.info("Created %s downloader successfully", downloader.get_service_name())

        # Step 2: Process all identifiers
        article_data = downloader.process_identifiers(identifiers)

        # Step 3: Build summary for user
        content = downloader.build_summary(article_data)

        # Step 4: Log results summary
        total_papers = len(article_data)
        successful_downloads = sum(
            1
            for paper in article_data.values()
            if paper.get("access_type") == "open_access_downloaded"
        )
        logger.info(
            "Download complete for %s: %d papers processed, %d PDFs downloaded",
            service,
            total_papers,
            successful_downloads,
        )

        # Step 5: Return command with results
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

    except ValueError as e:
        # Handle service/configuration errors
        error_msg = f"Service error for '{service}': {str(e)}"
        logger.error(error_msg)

        return Command(
            update={
                "article_data": {},
                "messages": [
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call_id,
                        artifact={},
                    )
                ],
            }
        )

    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error during paper download: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return Command(
            update={
                "article_data": {},
                "messages": [
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call_id,
                        artifact={},
                    )
                ],
            }
        )
