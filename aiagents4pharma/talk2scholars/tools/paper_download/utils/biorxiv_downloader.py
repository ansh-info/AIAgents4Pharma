#!/usr/bin/env python3
"""
BioRxiv paper downloader implementation.
"""

import logging
import tempfile
from typing import Any, Dict, Optional, Tuple

import cloudscraper

from .base_paper_downloader import BasePaperDownloader

logger = logging.getLogger(__name__)


class BiorxivDownloader(BasePaperDownloader):
    """BioRxiv-specific implementation of paper downloader."""

    def __init__(self, config: Any):
        """Initialize BioRxiv downloader with configuration."""
        super().__init__(config)
        self.api_url = config.api_url
        self.pdf_base_url = getattr(
            config, "pdf_base_url", "https://www.biorxiv.org/content/10.1101/"
        )
        self.landing_url_template = getattr(
            config,
            "landing_url_template",
            "https://www.biorxiv.org/content/{doi}v{version}",
        )
        self.pdf_url_template = getattr(
            config,
            "pdf_url_template",
            "https://www.biorxiv.org/content/{doi}v{version}.full.pdf",
        )

        # CloudScraper specific settings
        self.cf_clearance_timeout = getattr(config, "cf_clearance_timeout", 30)
        self.session_reuse = getattr(config, "session_reuse", True)

        # Initialize shared CloudScraper session if enabled
        self._scraper = None
        if self.session_reuse:
            self._scraper = cloudscraper.create_scraper(
                browser={"custom": self.user_agent}, delay=self.cf_clearance_timeout
            )

    def fetch_metadata(self, identifier: str) -> Dict[str, Any]:
        """
        Fetch paper metadata from bioRxiv API.

        Args:
            identifier: DOI (e.g., '10.1101/2020.09.09.20191205')

        Returns:
            JSON response as dictionary from bioRxiv API

        Raises:
            requests.RequestException: If API call fails
            RuntimeError: If no collection data found in response
        """
        query_url = f"{self.api_url}/biorxiv/{identifier}/na/json"
        logger.info("Fetching metadata for DOI %s from: %s", identifier, query_url)

        # Use CloudScraper for metadata as well, in case API is behind CF protection
        scraper = self._scraper or cloudscraper.create_scraper(
            browser={"custom": self.user_agent}, delay=self.cf_clearance_timeout
        )

        response = scraper.get(query_url, timeout=self.request_timeout)
        response.raise_for_status()

        paper_data = response.json()

        if "collection" not in paper_data or not paper_data["collection"]:
            raise RuntimeError("No collection data found in bioRxiv API response")

        return paper_data

    def construct_pdf_url(self, metadata: Dict[str, Any], identifier: str) -> str:
        """
        Construct PDF URL from bioRxiv metadata and DOI.

        Args:
            metadata: JSON response from bioRxiv API
            identifier: DOI

        Returns:
            Constructed PDF URL string
        """
        if "collection" not in metadata or not metadata["collection"]:
            return ""

        paper = metadata["collection"][0]  # Get first (and should be only) paper
        version = paper.get("version", "1")  # Default to version 1

        # Construct bioRxiv PDF URL using template
        pdf_url = self.pdf_url_template.format(doi=identifier, version=version)
        logger.info("Constructed PDF URL for DOI %s: %s", identifier, pdf_url)

        return pdf_url

    def download_pdf_to_temp(
        self, pdf_url: str, identifier: str
    ) -> Optional[Tuple[str, str]]:
        """
        Override base method to use CloudScraper for bioRxiv PDF downloads.
        Includes landing page visit to handle CloudFlare protection.

        Args:
            pdf_url: URL to download PDF from
            identifier: DOI for logging

        Returns:
            Tuple of (temp_file_path, filename) or None if failed
        """
        if not pdf_url:
            logger.info("No PDF URL available for DOI %s", identifier)
            return None

        try:
            logger.info("Downloading PDF for DOI %s from %s", identifier, pdf_url)

            # Get or create scraper for this download
            scraper = self._scraper or cloudscraper.create_scraper(
                browser={"custom": self.user_agent}, delay=self.cf_clearance_timeout
            )

            # Extract version from PDF URL to construct landing page URL
            # PDF URL format: https://www.biorxiv.org/content/{doi}v{version}.full.pdf
            # Landing URL format: https://www.biorxiv.org/content/{doi}v{version}
            if ".full.pdf" in pdf_url:
                landing_url = pdf_url.replace(".full.pdf", "")
                logger.info("Visiting landing page first: %s", landing_url)

                # Step 1: Visit landing page to solve CF JS-challenge
                landing_response = scraper.get(
                    landing_url, timeout=self.request_timeout
                )
                landing_response.raise_for_status()
                logger.info("Successfully accessed landing page for %s", identifier)

            # Step 2: Download the PDF using the same session
            response = scraper.get(pdf_url, timeout=self.request_timeout, stream=True)
            response.raise_for_status()

            # Step 3: Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        temp_file.write(chunk)
                temp_file_path = temp_file.name

            logger.info("BioRxiv PDF downloaded to temporary file: %s", temp_file_path)

            # Generate filename from DOI
            filename = self.get_default_filename(identifier)

            # Try to extract filename from Content-Disposition header
            content_disposition = response.headers.get("Content-Disposition", "")
            if "filename=" in content_disposition:
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
                    logger.warning("Failed to extract filename from header: %s", e)

            return temp_file_path, filename

        except Exception as e:
            logger.error("Failed to download PDF for DOI %s: %s", identifier, e)
            return None

    def extract_paper_metadata(
        self,
        metadata: Dict[str, Any],
        identifier: str,
        pdf_result: Optional[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """
        Extract structured metadata from bioRxiv API response.

        Args:
            metadata: JSON response from bioRxiv API
            identifier: DOI
            pdf_result: Tuple of (temp_file_path, filename) if PDF downloaded

        Returns:
            Standardized paper metadata dictionary
        """
        if "collection" not in metadata or not metadata["collection"]:
            raise RuntimeError("No collection data found in metadata")

        paper = metadata["collection"][0]  # Get first (and should be only) paper

        # Extract title
        title = paper.get("title", "N/A").strip()

        # Extract authors - typically in a semicolon-separated string
        authors_str = paper.get("authors", "")
        authors = (
            [author.strip() for author in authors_str.split(";") if author.strip()]
            if authors_str
            else []
        )

        # Extract abstract
        abstract = paper.get("abstract", "N/A").strip()

        # Extract publication date
        pub_date = paper.get("date", "N/A").strip()

        # Extract additional bioRxiv-specific fields
        category = paper.get("category", "N/A").strip()
        version = paper.get("version", "N/A")

        # Handle PDF download results
        if pdf_result:
            temp_file_path, filename = pdf_result
            pdf_url = temp_file_path  # Use local temp file path
            access_type = "open_access_downloaded"
        else:
            temp_file_path = ""
            filename = self.get_default_filename(identifier)
            pdf_url = ""
            access_type = "download_failed"

        return {
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Publication Date": pub_date,
            "DOI": identifier,
            "Category": category,
            "Version": version,
            "URL": pdf_url,
            "pdf_url": pdf_url,
            "filename": filename,
            "source": "biorxiv",
            "server": "biorxiv",
            "access_type": access_type,
            "temp_file_path": temp_file_path,
        }

    def get_service_name(self) -> str:
        """Return service name."""
        return "bioRxiv"

    def get_identifier_name(self) -> str:
        """Return identifier display name."""
        return "DOI"

    def get_default_filename(self, identifier: str) -> str:
        """Generate default filename for bioRxiv paper."""
        # Sanitize DOI for filename use
        return f"{identifier.replace('/', '_').replace('.', '_')}.pdf"

    def _get_paper_identifier_info(self, paper: Dict[str, Any]) -> str:
        """Get bioRxiv-specific identifier info for paper summary."""
        doi = paper.get("DOI", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        category = paper.get("Category", "N/A")

        info = f" (DOI:{doi}, {pub_date})"
        if category != "N/A":
            info += f"\n   Category: {category}"

        return info

    def _add_service_identifier(self, entry: Dict[str, Any], identifier: str) -> None:
        """Add DOI and bioRxiv-specific fields to entry."""
        entry["DOI"] = identifier
        entry["Category"] = "N/A"
        entry["Version"] = "N/A"
        entry["server"] = "biorxiv"
