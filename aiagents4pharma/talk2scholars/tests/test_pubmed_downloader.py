"""
Unit tests for PubmedDownloader.
Tests PMID to PMCID conversion, XML parsing, and PDF URL extraction from multiple sources.
"""

import unittest
from unittest.mock import Mock, patch

import requests

from aiagents4pharma.talk2scholars.tools.paper_download.utils.pubmed_downloader import (
    PubmedDownloader,
)


class TestPubmedDownloader(unittest.TestCase):
    """Tests for the PubmedDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.id_converter_url = (
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0"
        )
        self.mock_config.oa_api_url = (
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        )
        self.mock_config.europe_pmc_base_url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest"
        )
        self.mock_config.pmc_page_base_url = "https://www.ncbi.nlm.nih.gov/pmc/articles"
        self.mock_config.direct_pmc_pdf_base_url = (
            "https://www.ncbi.nlm.nih.gov/pmc/articles"
        )
        self.mock_config.ftp_base_url = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc"
        self.mock_config.https_base_url = "https://www.ncbi.nlm.nih.gov/pmc"
        self.mock_config.user_agent = "Mozilla/5.0 (compatible; test-agent)"
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192

        self.downloader = PubmedDownloader(self.mock_config)

    def test_initialization(self):
        """Test PubmedDownloader initialization."""
        self.assertEqual(
            self.downloader.id_converter_url,
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0",
        )
        self.assertEqual(
            self.downloader.oa_api_url,
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
        )
        self.assertEqual(
            self.downloader.europe_pmc_base_url,
            "https://www.ebi.ac.uk/europepmc/webservices/rest",
        )
        self.assertEqual(
            self.downloader.pmc_page_base_url,
            "https://www.ncbi.nlm.nih.gov/pmc/articles",
        )
        self.assertEqual(
            self.downloader.direct_pmc_pdf_base_url,
            "https://www.ncbi.nlm.nih.gov/pmc/articles",
        )

    @patch("requests.get")
    def test_fetch_metadata_success(self, mock_get):
        """Test successful metadata fetching from ID converter API."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "records": [
                {"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1234/test"}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader.fetch_metadata("12345678")

        # Verify API call
        expected_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0?ids=12345678&format=json"
        mock_get.assert_called_once_with(expected_url, timeout=30)

        # Verify response structure
        self.assertIn("records", result)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["pmcid"], "PMC123456")

    @patch("requests.get")
    def test_fetch_metadata_no_records(self, mock_get):
        """Test fetch_metadata when no records found."""
        mock_response = Mock()
        mock_response.json.return_value = {"records": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as context:
            self.downloader.fetch_metadata("12345678")

        self.assertIn("No records found", str(context.exception))

    @patch("requests.get")
    def test_fetch_metadata_network_error(self, mock_get):
        """Test fetch_metadata with network error."""
        mock_get.side_effect = requests.RequestException("Network error")

        with self.assertRaises(requests.RequestException):
            self.downloader.fetch_metadata("12345678")

    @patch("requests.get")
    def test_try_oa_api_success(self, mock_get):
        """Test successful OA API PDF URL extraction."""
        mock_response = Mock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <records>
                <record>
                    <link format="pdf" href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf"/>
                </record>
            </records>
        </OA>"""
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader._try_oa_api("PMC123456")

        expected_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC123456"
        mock_get.assert_called_once_with(expected_url, timeout=30)

        # Should return the URL from the XML
        self.assertEqual(
            result, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf"
        )
        self.assertIn("PMC123456", result)

    @patch("requests.get")
    def test_try_oa_api_error_response(self, mock_get):
        """Test OA API with error response."""
        mock_response = Mock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <error code="idDoesNotExist">Invalid PMC ID</error>
        </OA>"""
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader._try_oa_api("PMC123456")

        self.assertEqual(result, "")

    @patch("requests.get")
    def test_try_oa_api_network_error(self, mock_get):
        """Test OA API with network error."""
        mock_get.side_effect = requests.RequestException("Network error")

        result = self.downloader._try_oa_api("PMC123456")

        self.assertEqual(result, "")

    @patch("requests.head")
    def test_try_europe_pmc_success(self, mock_head):
        """Test successful Europe PMC PDF access."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        result = self.downloader._try_europe_pmc("PMC123456")

        expected_url = "https://www.ebi.ac.uk/europepmc/webservices/rest?accid=PMC123456&blobtype=pdf"
        mock_head.assert_called_once_with(expected_url, timeout=30)
        self.assertEqual(result, expected_url)

    @patch("requests.head")
    def test_try_europe_pmc_not_found(self, mock_head):
        """Test Europe PMC when PDF not available."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response

        result = self.downloader._try_europe_pmc("PMC123456")

        self.assertEqual(result, "")

    @patch("requests.head")
    def test_try_europe_pmc_network_error(self, mock_head):
        """Test Europe PMC with network error."""
        mock_head.side_effect = requests.RequestException("Network error")

        result = self.downloader._try_europe_pmc("PMC123456")

        self.assertEqual(result, "")

    @patch("requests.get")
    def test_try_pmc_page_scraping_success(self, mock_get):
        """Test successful PMC page scraping for PDF URL."""
        mock_response = Mock()
        html_content = """
        <html>
            <head>
                <meta name="citation_pdf_url" content="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf">
            </head>
        </html>"""
        mock_response.content = html_content.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader._try_pmc_page_scraping("PMC123456")

        expected_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/"
        expected_headers = {"User-Agent": "Mozilla/5.0 (compatible; test-agent)"}
        mock_get.assert_called_once_with(
            expected_url, headers=expected_headers, timeout=30
        )
        self.assertEqual(
            result, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf"
        )

    @patch("requests.get")
    def test_try_pmc_page_scraping_no_pdf(self, mock_get):
        """Test PMC page scraping when no PDF URL found."""
        mock_response = Mock()
        mock_response.content = "<html><head></head></html>".encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader._try_pmc_page_scraping("PMC123456")

        self.assertEqual(result, "")

    @patch("requests.get")
    def test_try_pmc_page_scraping_network_error(self, mock_get):
        """Test PMC page scraping with network error."""
        mock_get.side_effect = requests.RequestException("Network error")

        result = self.downloader._try_pmc_page_scraping("PMC123456")

        self.assertEqual(result, "")

    @patch("requests.head")
    def test_try_direct_pmc_url_success(self, mock_head):
        """Test successful direct PMC PDF URL access."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        result = self.downloader._try_direct_pmc_url("PMC123456")

        expected_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"
        mock_head.assert_called_once_with(expected_url, timeout=30)
        self.assertEqual(result, expected_url)

    @patch("requests.head")
    def test_try_direct_pmc_url_not_found(self, mock_head):
        """Test direct PMC URL when not accessible."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response

        result = self.downloader._try_direct_pmc_url("PMC123456")

        self.assertEqual(result, "")

    def test_construct_pdf_url_success(self):
        """Test successful PDF URL construction."""
        metadata = {"records": [{"pmcid": "PMC123456", "doi": "10.1234/test"}]}

        with patch.object(
            self.downloader,
            "_fetch_pdf_url_with_fallbacks",
            return_value="http://test.pdf",
        ) as mock_fetch:
            result = self.downloader.construct_pdf_url(metadata, "12345678")

        self.assertEqual(result, "http://test.pdf")
        mock_fetch.assert_called_once_with("PMC123456")

    def test_construct_pdf_url_no_records(self):
        """Test PDF URL construction with no records."""
        metadata = {"records": []}

        result = self.downloader.construct_pdf_url(metadata, "12345678")

        self.assertEqual(result, "")

    def test_construct_pdf_url_no_pmcid(self):
        """Test PDF URL construction with no PMCID."""
        metadata = {"records": [{"pmcid": "N/A", "doi": "10.1234/test"}]}

        result = self.downloader.construct_pdf_url(metadata, "12345678")

        self.assertEqual(result, "")

    def test_fetch_pdf_url_with_fallbacks_multiple_sources(self):
        """Test _fetch_pdf_url_with_fallbacks trying multiple sources."""
        with patch.object(self.downloader, "_try_oa_api", return_value="") as mock_oa:
            with patch.object(
                self.downloader, "_try_europe_pmc", return_value=""
            ) as mock_europe:
                with patch.object(
                    self.downloader,
                    "_try_pmc_page_scraping",
                    return_value="http://test.pdf",
                ) as mock_scrape:
                    with patch.object(
                        self.downloader, "_try_direct_pmc_url", return_value=""
                    ) as mock_direct:
                        result = self.downloader._fetch_pdf_url_with_fallbacks(
                            "PMC123456"
                        )

        self.assertEqual(result, "http://test.pdf")
        mock_oa.assert_called_once_with("PMC123456")
        mock_europe.assert_called_once_with("PMC123456")
        mock_scrape.assert_called_once_with("PMC123456")
        # mock_direct should not be called since scraping succeeded
        mock_direct.assert_not_called()

    def test_extract_paper_metadata_success(self):
        """Test successful paper metadata extraction."""
        metadata = {
            "records": [
                {"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1234/test"}
            ]
        }
        pdf_result = ("/tmp/paper.pdf", "paper.pdf")

        result = self.downloader.extract_paper_metadata(
            metadata, "12345678", pdf_result
        )

        expected = {
            "Title": "PubMed Article 12345678",
            "Authors": [],
            "Abstract": "Abstract available in PubMed",
            "Publication Date": "N/A",
            "PMID": "12345678",
            "PMCID": "PMC123456",
            "DOI": "10.1234/test",
            "Journal": "N/A",
            "URL": "/tmp/paper.pdf",
            "pdf_url": "/tmp/paper.pdf",
            "filename": "paper.pdf",
            "source": "pubmed",
            "access_type": "open_access_downloaded",
            "temp_file_path": "/tmp/paper.pdf",
        }

        self.assertEqual(result, expected)

    def test_extract_paper_metadata_no_pdf(self):
        """Test metadata extraction without PDF download."""
        metadata = {
            "records": [
                {"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1234/test"}
            ]
        }

        result = self.downloader.extract_paper_metadata(metadata, "12345678", None)

        self.assertEqual(result["Title"], "PubMed Article 12345678")
        self.assertEqual(result["URL"], "")
        self.assertEqual(result["access_type"], "abstract_only")
        self.assertEqual(result["filename"], "pmid_12345678.pdf")

    def test_get_service_name(self):
        """Test get_service_name method."""
        result = self.downloader.get_service_name()
        self.assertEqual(result, "PubMed")

    def test_get_identifier_name(self):
        """Test get_identifier_name method."""
        result = self.downloader.get_identifier_name()
        self.assertEqual(result, "PMID")

    def test_get_default_filename(self):
        """Test get_default_filename method."""
        result = self.downloader.get_default_filename("12345678")
        self.assertEqual(result, "pmid_12345678.pdf")

    def test_get_paper_identifier_info(self):
        """Test _get_paper_identifier_info method."""
        paper = {"PMID": "12345678", "PMCID": "PMC123456"}

        result = self.downloader._get_paper_identifier_info(paper)

        self.assertIn("12345678", result)
        self.assertIn("PMC123456", result)

    def test_add_service_identifier(self):
        """Test _add_service_identifier method."""
        entry = {}

        self.downloader._add_service_identifier(entry, "12345678")

        self.assertEqual(entry["PMID"], "12345678")
        self.assertEqual(entry["PMCID"], "N/A")
        self.assertEqual(entry["DOI"], "N/A")
        self.assertEqual(entry["Journal"], "N/A")

    def test_get_snippet(self):
        """Test get_snippet method with PubMed-specific handling."""
        # Test with standard abstract
        result = self.downloader.get_snippet(
            "This is a normal abstract that should be returned."
        )
        self.assertEqual(result, "This is a normal abstract that should be returned.")

        # Test with empty abstract
        result = self.downloader.get_snippet("")
        self.assertEqual(result, "")

        # Test with N/A abstract
        result = self.downloader.get_snippet("N/A")
        self.assertEqual(result, "")

        # Test with PubMed placeholder abstract
        result = self.downloader.get_snippet("Abstract available in PubMed")
        self.assertEqual(result, "")

    def test_extract_paper_metadata_no_pmcid(self):
        """Test metadata extraction with no PMCID."""
        metadata = {
            "records": [{"pmid": "12345678", "pmcid": "N/A", "doi": "10.1234/test"}]
        }

        result = self.downloader.extract_paper_metadata(metadata, "12345678", None)

        self.assertEqual(result["access_type"], "no_pmcid")
        self.assertEqual(result["PMCID"], "N/A")

    def test_extract_paper_metadata_empty_records(self):
        """Test metadata extraction with empty records list."""
        metadata = {"records": []}

        with self.assertRaises(RuntimeError) as context:
            self.downloader.extract_paper_metadata(metadata, "12345678", None)

        self.assertIn("No records found in metadata", str(context.exception))

    def test_extract_paper_metadata_missing_records_key(self):
        """Test metadata extraction with missing records key."""
        metadata = {"other_key": "value"}

        with self.assertRaises(RuntimeError) as context:
            self.downloader.extract_paper_metadata(metadata, "12345678", None)

        self.assertIn("No records found in metadata", str(context.exception))

    def test_ftp_to_https_conversion(self):
        """Test FTP to HTTPS URL conversion in OA API response."""
        mock_response_text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <records>
                <record>
                    <link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/test.pdf"/>
                </record>
            </records>
        </OA>"""

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = mock_response_text
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = self.downloader._try_oa_api("PMC123456")

            # Should convert FTP to HTTPS
            self.assertTrue(result.startswith("https://www.ncbi.nlm.nih.gov/pmc"))
            self.assertNotIn("ftp://", result)
            self.assertIn("test.pdf", result)

    def test_fetch_pdf_url_with_fallbacks_europe_pmc_success(self):
        """Test _fetch_pdf_url_with_fallbacks with Europe PMC success."""
        with patch.object(self.downloader, "_try_oa_api", return_value="") as mock_oa:
            with patch.object(
                self.downloader, "_try_europe_pmc", return_value="http://europe.pdf"
            ) as mock_europe:
                with patch.object(
                    self.downloader, "_try_pmc_page_scraping"
                ) as mock_scrape:
                    with patch.object(
                        self.downloader, "_try_direct_pmc_url"
                    ) as mock_direct:
                        result = self.downloader._fetch_pdf_url_with_fallbacks(
                            "PMC123456"
                        )

        self.assertEqual(result, "http://europe.pdf")
        mock_oa.assert_called_once_with("PMC123456")
        mock_europe.assert_called_once_with("PMC123456")
        # Should stop at Europe PMC, not try further methods
        mock_scrape.assert_not_called()
        mock_direct.assert_not_called()

    def test_fetch_pdf_url_with_fallbacks_direct_pmc_success(self):
        """Test _fetch_pdf_url_with_fallbacks with Direct PMC success."""
        with patch.object(self.downloader, "_try_oa_api", return_value="") as mock_oa:
            with patch.object(
                self.downloader, "_try_europe_pmc", return_value=""
            ) as mock_europe:
                with patch.object(
                    self.downloader, "_try_pmc_page_scraping", return_value=""
                ) as mock_scrape:
                    with patch.object(
                        self.downloader,
                        "_try_direct_pmc_url",
                        return_value="http://direct.pdf",
                    ) as mock_direct:
                        result = self.downloader._fetch_pdf_url_with_fallbacks(
                            "PMC123456"
                        )

        self.assertEqual(result, "http://direct.pdf")
        mock_oa.assert_called_once_with("PMC123456")
        mock_europe.assert_called_once_with("PMC123456")
        mock_scrape.assert_called_once_with("PMC123456")
        mock_direct.assert_called_once_with("PMC123456")

    def test_fetch_pdf_url_with_fallbacks_all_fail(self):
        """Test _fetch_pdf_url_with_fallbacks when all strategies fail."""
        with patch.object(self.downloader, "_try_oa_api", return_value="") as mock_oa:
            with patch.object(
                self.downloader, "_try_europe_pmc", return_value=""
            ) as mock_europe:
                with patch.object(
                    self.downloader, "_try_pmc_page_scraping", return_value=""
                ) as mock_scrape:
                    with patch.object(
                        self.downloader, "_try_direct_pmc_url", return_value=""
                    ) as mock_direct:
                        result = self.downloader._fetch_pdf_url_with_fallbacks(
                            "PMC123456"
                        )

        self.assertEqual(result, "")
        mock_oa.assert_called_once_with("PMC123456")
        mock_europe.assert_called_once_with("PMC123456")
        mock_scrape.assert_called_once_with("PMC123456")
        mock_direct.assert_called_once_with("PMC123456")

    @patch("requests.head")
    def test_try_direct_pmc_url_exception(self, mock_head):
        """Test _try_direct_pmc_url with network exception."""
        mock_head.side_effect = requests.RequestException("Network error")

        result = self.downloader._try_direct_pmc_url("PMC123456")

        self.assertEqual(result, "")
        expected_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"
        mock_head.assert_called_once_with(expected_url, timeout=30)


class TestPubmedDownloaderIntegration(unittest.TestCase):
    """Integration tests for PubmedDownloader workflow."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_config = Mock()
        self.mock_config.id_converter_url = (
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0"
        )
        self.mock_config.oa_api_url = (
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        )
        self.mock_config.europe_pmc_base_url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest"
        )
        self.mock_config.pmc_page_base_url = "https://www.ncbi.nlm.nih.gov/pmc/articles"
        self.mock_config.direct_pmc_pdf_base_url = (
            "https://www.ncbi.nlm.nih.gov/pmc/articles"
        )
        self.mock_config.ftp_base_url = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc"
        self.mock_config.https_base_url = "https://www.ncbi.nlm.nih.gov/pmc"
        self.mock_config.user_agent = "Mozilla/5.0 (compatible; test-agent)"
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192

        self.downloader = PubmedDownloader(self.mock_config)

    @patch("requests.get")
    def test_full_workflow_pmid_to_pdf(self, mock_get):
        """Test complete workflow from PMID to PDF download."""
        # Mock ID converter response
        metadata_response = Mock()
        metadata_response.json.return_value = {
            "records": [
                {"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1234/test"}
            ]
        }
        metadata_response.raise_for_status = Mock()

        # Mock OA API response
        oa_response = Mock()
        oa_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <records>
                <record>
                    <link format="pdf" href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf"/>
                </record>
            </records>
        </OA>"""
        oa_response.raise_for_status = Mock()

        # We need to mock multiple requests: metadata + OA API + potential fallbacks
        mock_get.return_value = metadata_response  # Default for metadata fetch

        # Override for specific calls
        def get_side_effect(url, **kwargs):
            if "idconv" in url:
                return metadata_response
            elif "oa.fcgi" in url:
                return oa_response
            # No fallback needed - test only uses these two URLs

        mock_get.side_effect = get_side_effect

        # Simulate the workflow
        identifier = "12345678"

        # Step 1: Fetch metadata
        metadata = self.downloader.fetch_metadata(identifier)

        # Step 2: Construct PDF URL (tries multiple sources)
        pdf_url = self.downloader.construct_pdf_url(metadata, identifier)

        # Verify results
        self.assertEqual(metadata["records"][0]["pmid"], "12345678")
        self.assertEqual(metadata["records"][0]["pmcid"], "PMC123456")
        self.assertIn("PMC123456", pdf_url)
        self.assertTrue(pdf_url.startswith("https://"))

        # Verify API calls
        self.assertEqual(mock_get.call_count, 2)

        # First call should be ID converter
        first_call = mock_get.call_args_list[0]
        self.assertIn("idconv", first_call[0][0])

        # Second call should be OA API
        second_call = mock_get.call_args_list[1]
        self.assertIn("oa.fcgi", second_call[0][0])

    @patch("requests.get")
    def test_workflow_with_fallback_sources(self, mock_get):
        """Test workflow that falls back through multiple PDF sources."""
        # Mock ID converter response
        metadata_response = Mock()
        metadata_response.json.return_value = {
            "records": [
                {"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1234/test"}
            ]
        }
        metadata_response.raise_for_status = Mock()

        # Mock OA API failure
        oa_response = Mock()
        oa_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <error code="idDoesNotExist">Invalid PMC ID</error>
        </OA>"""
        oa_response.raise_for_status = Mock()

        # Mock scraping success
        scrape_response = Mock()
        html_content = """
        <html>
            <head>
                <meta name="citation_pdf_url" content="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/fallback.pdf">
            </head>
        </html>"""
        scrape_response.content = html_content.encode()
        scrape_response.raise_for_status = Mock()

        mock_get.side_effect = [metadata_response, oa_response, scrape_response]

        with patch("requests.head") as mock_head:
            # Mock Europe PMC failure
            mock_head.return_value.status_code = 404

            # Run workflow
            identifier = "12345678"
            metadata = self.downloader.fetch_metadata(identifier)
            pdf_url = self.downloader.construct_pdf_url(metadata, identifier)

        # Should have fallen back to scraping
        self.assertEqual(
            pdf_url,
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/fallback.pdf",
        )

        # Verify all sources were tried
        self.assertEqual(mock_get.call_count, 3)  # ID converter + OA API + scraping
        mock_head.assert_called_once()  # Europe PMC
