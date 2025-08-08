"""
Unit tests for PubmedDownloader.
Tests PMID to PMCID conversion, XML parsing, and PDF URL extraction
from multiple sources. Uses a public shim to avoid accessing protected
members in tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from aiagents4pharma.talk2scholars.tools.paper_download.utils.pubmed_downloader import (
    PubmedDownloader,
)


class PubmedDownloaderTestShim(PubmedDownloader):
    """Thin public shim that forwards to the real implementation."""

    __test__ = False  # prevent pytest from collecting it as a test

    # Public wrappers for protected helpers
    def try_oa_api_public(self, pmcid: str) -> str:
        """Public wrapper for _try_oa_api method."""
        return self._try_oa_api(pmcid)

    def try_europe_pmc_public(self, pmcid: str) -> str:
        """Public wrapper for _try_europe_pmc method."""
        return self._try_europe_pmc(pmcid)

    def try_pmc_page_scraping_public(self, pmcid: str) -> str:
        """Public wrapper for _try_pmc_page_scraping method."""
        return self._try_pmc_page_scraping(pmcid)

    def try_direct_pmc_url_public(self, pmcid: str) -> str:
        """Public wrapper for _try_direct_pmc_url method."""
        return self._try_direct_pmc_url(pmcid)

    def fetch_pdf_url_with_fallbacks_public(self, pmcid: str) -> str:
        """Same fallback order as production, but via public wrappers."""
        for fn in (
            self.try_oa_api_public,
            self.try_europe_pmc_public,
            self.try_pmc_page_scraping_public,
            self.try_direct_pmc_url_public,
        ):
            url = fn(pmcid)
            if url:
                return url
        return ""

    # IMPORTANT: override to use the shim's public chain so tests can patch it
    def construct_pdf_url(self, metadata, identifier):  # same signature
        """Test-friendly override that uses the shim's public fallback chain."""
        if "records" not in metadata or not metadata["records"]:
            return ""
        pmcid = metadata["records"][0].get("pmcid", "")
        if not pmcid or pmcid == "N/A":
            return ""
        return self.fetch_pdf_url_with_fallbacks_public(pmcid)

    # Public accessors for identifier helpers (avoid protected-access in tests)
    def get_paper_identifier_info_public(self, paper):
        """Public wrapper for _get_paper_identifier_info method."""
        return self._get_paper_identifier_info(paper)

    def add_service_identifier_public(self, entry, identifier):
        """Public wrapper for _add_service_identifier method."""
        return self._add_service_identifier(entry, identifier)


class TestPubmedDownloaderBasics(unittest.TestCase):
    """Basic metadata + OA API parsing tests (hit the production code)."""

    def setUp(self):
        cfg = SimpleNamespace(
            id_converter_url="https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0",
            oa_api_url="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            europe_pmc_base_url="https://www.ebi.ac.uk/europepmc/webservices/rest",
            pmc_page_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            direct_pmc_pdf_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            ftp_base_url="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc",
            https_base_url="https://www.ncbi.nlm.nih.gov/pmc",
            user_agent="Mozilla/5.0 (compatible; test-agent)",
            request_timeout=30,
            chunk_size=8192,
        )
        self.downloader = PubmedDownloaderTestShim(cfg)

    def test_initialization(self):
        """Sanity check config wiring."""
        self.assertIn("idconv", self.downloader.id_converter_url)
        self.assertIn("oa.fcgi", self.downloader.oa_api_url)

    @patch("requests.get")
    def test_fetch_metadata_success(self, mock_get):
        """Successful PMID→PMCID conversion."""
        resp = Mock()
        resp.json.return_value = {
            "records": [{"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1/x"}]
        }
        resp.raise_for_status = Mock()
        mock_get.return_value = resp

        data = self.downloader.fetch_metadata("12345678")
        mock_get.assert_called_once()
        self.assertIn("records", data)
        self.assertEqual(data["records"][0]["pmcid"], "PMC123456")

    @patch("requests.get")
    def test_fetch_metadata_no_records(self, mock_get):
        """Test fetch_metadata with empty records."""
        resp = Mock()
        resp.json.return_value = {"records": []}
        resp.raise_for_status = Mock()
        mock_get.return_value = resp
        with self.assertRaises(RuntimeError):
            self.downloader.fetch_metadata("12345678")

    @patch("requests.get")
    def test_fetch_metadata_network_error(self, mock_get):
        """Test fetch_metadata with network error."""
        mock_get.side_effect = requests.RequestException("down")
        with self.assertRaises(requests.RequestException):
            self.downloader.fetch_metadata("12345678")

    # ---- OA API paths (cover lines ~77–87, 99–122) ----

    @patch("requests.get")
    def test_oa_api_xml_error_node_returns_empty(self, mock_get):
        """<error> node -> return empty string."""
        resp = Mock()
        resp.text = (
            '<?xml version="1.0"?><OA><error code="idDoesNotExist">'
            "Invalid PMC ID</error></OA>"
        )
        resp.raise_for_status = Mock()
        mock_get.return_value = resp

        out = self.downloader.try_oa_api_public("PMC999999")
        self.assertEqual(out, "")

    @patch("requests.get")
    def test_oa_api_pdf_link_success(self, mock_get):
        """<link format='pdf' href='https://...'> -> return the https link."""
        resp = Mock()
        resp.text = (
            '<?xml version="1.0"?><OA><records><record>'
            '<link format="pdf" '
            'href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/pdf/a.pdf"/>'
            "</record></records></OA>"
        )
        resp.raise_for_status = Mock()
        mock_get.return_value = resp

        out = self.downloader.try_oa_api_public("PMC1")
        self.assertTrue(out.endswith("/PMC1/pdf/a.pdf"))

    @patch("requests.get")
    def test_oa_api_ftp_link_converts_to_https(self, mock_get):
        """ftp:// link -> converted to https:// base (covers conversion branch)."""
        resp = Mock()
        resp.text = (
            '<?xml version="1.0"?><OA><records><record>'
            '<link format="pdf" '
            'href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/a/b/c.pdf"/>'
            "</record></records></OA>"
        )
        resp.raise_for_status = Mock()
        mock_get.return_value = resp

        out = self.downloader.try_oa_api_public("PMC2")
        self.assertTrue(out.startswith("https://www.ncbi.nlm.nih.gov/pmc"))
        self.assertTrue(out.endswith("c.pdf"))

    @patch("requests.get")
    def test_oa_api_network_exception_returns_empty(self, mock_get):
        """Test OA API with network exception returns empty string."""
        mock_get.side_effect = requests.RequestException("net")
        out = self.downloader.try_oa_api_public("PMC3")
        self.assertEqual(out, "")


# ------------------------ OA API ----------------------------------------------------
class TestPubmedDownloaderOAAPI(unittest.TestCase):
    """Tests for OA API and FTP->HTTPS conversion."""

    def setUp(self):
        cfg = SimpleNamespace(
            id_converter_url="",
            oa_api_url="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            europe_pmc_base_url="",
            pmc_page_base_url="",
            direct_pmc_pdf_base_url="",
            ftp_base_url="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc",
            https_base_url="https://www.ncbi.nlm.nih.gov/pmc",
            user_agent="Mozilla/5.0 (compatible; test-agent)",
            request_timeout=30,
            chunk_size=8192,
        )
        self.downloader = PubmedDownloaderTestShim(cfg)

    @patch("requests.get")
    def test_try_oa_api_success(self, mock_get):
        """Test successful OA API response."""
        mock_response = Mock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <records>
                <record>
                    <link format="pdf"
                     href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf"/>
                </record>
            </records>
        </OA>"""
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader.try_oa_api_public("PMC123456")
        expected_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC123456"
        mock_get.assert_called_once_with(expected_url, timeout=30)
        self.assertEqual(
            result,
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf",
        )
        self.assertIn("PMC123456", result)

    @patch("requests.get")
    def test_try_oa_api_error_response(self, mock_get):
        """Test OA API error response."""
        mock_response = Mock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <error code="idDoesNotExist">Invalid PMC ID</error>
        </OA>"""
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        result = self.downloader.try_oa_api_public("PMC123456")
        self.assertEqual(result, "")

    @patch("requests.get")
    def test_try_oa_api_network_error(self, mock_get):
        """Test OA API with network error."""
        mock_get.side_effect = requests.RequestException("Network error")
        result = self.downloader.try_oa_api_public("PMC123456")
        self.assertEqual(result, "")

    def test_ftp_to_https_conversion(self):
        """Test FTP to HTTPS URL conversion."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OA>
            <records>
                <record>
                    <link format="pdf"
                     href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/test.pdf"/>
                </record>
            </records>
        </OA>"""
        with patch("requests.get") as mock_get:
            resp = Mock()
            resp.text = xml
            resp.raise_for_status = Mock()
            mock_get.return_value = resp
            result = self.downloader.try_oa_api_public("PMC123456")
        self.assertTrue(result.startswith("https://www.ncbi.nlm.nih.gov/pmc"))
        self.assertNotIn("ftp://", result)
        self.assertIn("test.pdf", result)


# ------------------------ Europe PMC ------------------------------------------------
class TestPubmedDownloaderEuropePMC(unittest.TestCase):
    """Europe PMC link checking."""

    def setUp(self):
        cfg = SimpleNamespace(
            id_converter_url="",
            oa_api_url="",
            europe_pmc_base_url="https://www.ebi.ac.uk/europepmc/webservices/rest",
            pmc_page_base_url="",
            direct_pmc_pdf_base_url="",
            ftp_base_url="",
            https_base_url="",
            user_agent="Mozilla/5.0 (compatible; test-agent)",
            request_timeout=30,
            chunk_size=8192,
        )
        self.downloader = PubmedDownloaderTestShim(cfg)

    @patch("requests.head")
    def test_try_europe_pmc_success(self, mock_head):
        """Test successful Europe PMC response."""
        resp = Mock()
        resp.status_code = 200
        mock_head.return_value = resp
        result = self.downloader.try_europe_pmc_public("PMC123456")
        expected = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest"
            "?accid=PMC123456&blobtype=pdf"
        )
        mock_head.assert_called_once_with(expected, timeout=30)
        self.assertEqual(result, expected)

    @patch("requests.head")
    def test_try_europe_pmc_not_found(self, mock_head):
        """Test Europe PMC not found response."""
        resp = Mock()
        resp.status_code = 404
        mock_head.return_value = resp
        self.assertEqual(self.downloader.try_europe_pmc_public("PMC123456"), "")

    @patch("requests.head")
    def test_try_europe_pmc_network_error(self, mock_head):
        """Test Europe PMC with network error."""
        mock_head.side_effect = requests.RequestException("Network error")
        self.assertEqual(self.downloader.try_europe_pmc_public("PMC123456"), "")


# ------------------------ PMC Page Scraping ----------------------------------------
class TestPubmedDownloaderPMCScrape(unittest.TestCase):
    """Scraping from PMC page."""

    def setUp(self):
        cfg = SimpleNamespace(
            id_converter_url="",
            oa_api_url="",
            europe_pmc_base_url="",
            pmc_page_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            direct_pmc_pdf_base_url="",
            ftp_base_url="",
            https_base_url="",
            user_agent="Mozilla/5.0 (compatible; test-agent)",
            request_timeout=30,
            chunk_size=8192,
        )
        self.downloader = PubmedDownloaderTestShim(cfg)

    @patch("requests.get")
    def test_try_pmc_page_scraping_success(self, mock_get):
        """Test successful PMC page scraping."""
        resp = Mock()
        html = (
            '<html><head><meta name="citation_pdf_url" '
            'content="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf">'
            "</head></html>"
        )
        resp.content = html.encode()
        resp.raise_for_status = Mock()
        mock_get.return_value = resp

        result = self.downloader.try_pmc_page_scraping_public("PMC123456")

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
        """Test PMC page scraping with no PDF found."""
        resp = Mock()
        resp.content = "<html><head></head></html>".encode()
        resp.raise_for_status = Mock()
        mock_get.return_value = resp
        self.assertEqual(self.downloader.try_pmc_page_scraping_public("PMC123456"), "")

    @patch("requests.get")
    def test_try_pmc_page_scraping_network_error(self, mock_get):
        """Test PMC page scraping with network error."""
        mock_get.side_effect = requests.RequestException("Network error")
        self.assertEqual(self.downloader.try_pmc_page_scraping_public("PMC123456"), "")


# ------------------------ Direct PMC PDF -------------------------------------------
class TestPubmedDownloaderDirectPMC(unittest.TestCase):
    """Direct PMC PDF attempts."""

    def setUp(self):
        cfg = SimpleNamespace(
            id_converter_url="",
            oa_api_url="",
            europe_pmc_base_url="",
            pmc_page_base_url="",
            direct_pmc_pdf_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            ftp_base_url="",
            https_base_url="",
            user_agent="Mozilla/5.0 (compatible; test-agent)",
            request_timeout=30,
            chunk_size=8192,
        )
        self.downloader = PubmedDownloaderTestShim(cfg)

    @patch("requests.head")
    def test_try_direct_pmc_url_success(self, mock_head):
        """Test successful direct PMC URL access."""
        resp = Mock()
        resp.status_code = 200
        mock_head.return_value = resp
        result = self.downloader.try_direct_pmc_url_public("PMC123456")
        expected = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"
        mock_head.assert_called_once_with(expected, timeout=30)
        self.assertEqual(result, expected)

    @patch("requests.head")
    def test_try_direct_pmc_url_not_found(self, mock_head):
        """Test direct PMC URL not found."""
        resp = Mock()
        resp.status_code = 404
        mock_head.return_value = resp
        self.assertEqual(self.downloader.try_direct_pmc_url_public("PMC123456"), "")

    @patch("requests.head")
    def test_try_direct_pmc_url_exception(self, mock_head):
        """Test direct PMC URL with exception."""
        mock_head.side_effect = requests.RequestException("Network error")
        result = self.downloader.try_direct_pmc_url_public("PMC123456")
        self.assertEqual(result, "")
        expected_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"
        mock_head.assert_called_once_with(expected_url, timeout=30)


class TestPubmedDownloaderConstructAndFallbacks(unittest.TestCase):
    """Construct URL + fallback chains via public wrappers (no protected access)."""

    def setUp(self):
        cfg = SimpleNamespace(
            id_converter_url="https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0",
            oa_api_url="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            europe_pmc_base_url="https://www.ebi.ac.uk/europepmc/webservices/rest",
            pmc_page_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            direct_pmc_pdf_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            ftp_base_url="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc",
            https_base_url="https://www.ncbi.nlm.nih.gov/pmc",
            user_agent="Mozilla/5.0 (compatible; test-agent)",
            request_timeout=30,
            chunk_size=8192,
        )
        self.downloader = PubmedDownloaderTestShim(cfg)

    def test_construct_pdf_url_success(self):
        """construct_pdf_url routes to the public fallback chain in tests."""
        metadata = {"records": [{"pmcid": "PMC123456", "doi": "10.1234/test"}]}
        with patch.object(
            self.downloader,
            "fetch_pdf_url_with_fallbacks_public",
            return_value="http://test.pdf",
        ) as mock_fetch:
            result = self.downloader.construct_pdf_url(metadata, "12345678")
        self.assertEqual(result, "http://test.pdf")
        mock_fetch.assert_called_once_with("PMC123456")

    def test_construct_pdf_url_no_records(self):
        """Test construct_pdf_url with no records."""
        self.assertEqual(self.downloader.construct_pdf_url({}, "x"), "")

    def test_construct_pdf_url_no_pmcid(self):
        """Test construct_pdf_url with no PMCID."""
        md = {"records": [{"pmcid": "N/A", "doi": "10.1/x"}]}
        self.assertEqual(self.downloader.construct_pdf_url(md, "x"), "")

    def test_fetch_pdf_url_with_fallbacks_europe_pmc_success(self):
        """Test fallback chain with Europe PMC success."""
        with (
            patch.object(self.downloader, "try_oa_api_public", return_value="") as m_oa,
            patch.object(
                self.downloader, "try_europe_pmc_public", return_value="http://eu.pdf"
            ) as m_eu,
            patch.object(self.downloader, "try_pmc_page_scraping_public") as m_scr,
            patch.object(self.downloader, "try_direct_pmc_url_public") as m_dir,
        ):
            out = self.downloader.fetch_pdf_url_with_fallbacks_public("PMC123456")
        self.assertEqual(out, "http://eu.pdf")
        m_oa.assert_called_once_with("PMC123456")
        m_eu.assert_called_once_with("PMC123456")
        m_scr.assert_not_called()
        m_dir.assert_not_called()

    def test_fetch_pdf_url_with_fallbacks_multiple_sources(self):
        """Test fallback chain through multiple sources."""
        with (
            patch.object(self.downloader, "try_oa_api_public", return_value="") as m_oa,
            patch.object(
                self.downloader, "try_europe_pmc_public", return_value=""
            ) as m_eu,
            patch.object(
                self.downloader,
                "try_pmc_page_scraping_public",
                return_value="http://test.pdf",
            ) as m_scr,
            patch.object(
                self.downloader, "try_direct_pmc_url_public", return_value=""
            ) as m_dir,
        ):
            out = self.downloader.fetch_pdf_url_with_fallbacks_public("PMC123456")
        self.assertEqual(out, "http://test.pdf")
        m_oa.assert_called_once_with("PMC123456")
        m_eu.assert_called_once_with("PMC123456")
        m_scr.assert_called_once_with("PMC123456")
        m_dir.assert_not_called()

    def test_fetch_pdf_url_with_fallbacks_direct_pmc_success(self):
        """Test fallback chain with direct PMC success."""
        with (
            patch.object(self.downloader, "try_oa_api_public", return_value="") as m_oa,
            patch.object(
                self.downloader, "try_europe_pmc_public", return_value=""
            ) as m_eu,
            patch.object(
                self.downloader, "try_pmc_page_scraping_public", return_value=""
            ) as m_scr,
            patch.object(
                self.downloader,
                "try_direct_pmc_url_public",
                return_value="http://direct.pdf",
            ) as m_dir,
        ):
            out = self.downloader.fetch_pdf_url_with_fallbacks_public("PMC123456")
        self.assertEqual(out, "http://direct.pdf")
        m_oa.assert_called_once_with("PMC123456")
        m_eu.assert_called_once_with("PMC123456")
        m_scr.assert_called_once_with("PMC123456")
        m_dir.assert_called_once_with("PMC123456")

    def test_fetch_pdf_url_with_fallbacks_all_fail(self):
        """Test fallback chain when all sources fail."""
        with (
            patch.object(self.downloader, "try_oa_api_public", return_value="") as m_oa,
            patch.object(
                self.downloader, "try_europe_pmc_public", return_value=""
            ) as m_eu,
            patch.object(
                self.downloader, "try_pmc_page_scraping_public", return_value=""
            ) as m_scr,
            patch.object(
                self.downloader, "try_direct_pmc_url_public", return_value=""
            ) as m_dir,
        ):
            out = self.downloader.fetch_pdf_url_with_fallbacks_public("PMC123456")
        self.assertEqual(out, "")
        m_oa.assert_called_once_with("PMC123456")
        m_eu.assert_called_once_with("PMC123456")
        m_scr.assert_called_once_with("PMC123456")
        m_dir.assert_called_once_with("PMC123456")

    def test_identifier_helper_wrappers(self):
        """Covers _get_paper_identifier_info and _add_service_identifier via wrappers."""
        paper = {"PMID": "12345678", "PMCID": "PMC9"}
        info = self.downloader.get_paper_identifier_info_public(paper)
        self.assertIn("PMID: 12345678", info)
        self.assertIn("PMCID: PMC9", info)

        entry = {}
        self.downloader.add_service_identifier_public(entry, "11122233")
        self.assertEqual(
            entry,
            {"PMID": "11122233", "PMCID": "N/A", "DOI": "N/A", "Journal": "N/A"},
        )


# ------------------------ Integration-ish paths ------------------------------------
class TestPubmedDownloaderIntegration(unittest.TestCase):
    """Integration tests for PubmedDownloader workflow."""

    def setUp(self):
        cfg = SimpleNamespace(
            id_converter_url="https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0",
            oa_api_url="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            europe_pmc_base_url="https://www.ebi.ac.uk/europepmc/webservices/rest",
            pmc_page_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            direct_pmc_pdf_base_url="https://www.ncbi.nlm.nih.gov/pmc/articles",
            ftp_base_url="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc",
            https_base_url="https://www.ncbi.nlm.nih.gov/pmc",
            user_agent="Mozilla/5.0 (compatible; test-agent)",
            request_timeout=30,
            chunk_size=8192,
        )
        self.downloader = PubmedDownloaderTestShim(cfg)

    @patch("requests.get")
    def test_full_workflow_pmid_to_pdf(self, mock_get):
        """Test full workflow from PMID to PDF URL."""
        metadata_response = Mock()
        metadata_response.json.return_value = {
            "records": [
                {"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1234/test"}
            ]
        }
        metadata_response.raise_for_status = Mock()

        oa_response = Mock()
        oa_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA><records><record>
        <link format="pdf"
         href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/test.pdf"/>
        </record></records></OA>"""
        oa_response.raise_for_status = Mock()

        def get_side_effect(url, *_, **__):
            if "idconv" in url:
                return metadata_response
            if "oa.fcgi" in url:
                return oa_response
            return None

        mock_get.side_effect = get_side_effect

        identifier = "12345678"
        metadata = self.downloader.fetch_metadata(identifier)
        pdf_url = self.downloader.construct_pdf_url(metadata, identifier)

        self.assertEqual(metadata["records"][0]["pmid"], "12345678")
        self.assertEqual(metadata["records"][0]["pmcid"], "PMC123456")
        self.assertIn("PMC123456", pdf_url)
        self.assertTrue(pdf_url.startswith("https://"))
        self.assertEqual(mock_get.call_count, 2)
        self.assertIn("idconv", mock_get.call_args_list[0][0][0])
        self.assertIn("oa.fcgi", mock_get.call_args_list[1][0][0])

    @patch("requests.get")
    def test_workflow_with_fallback_sources(self, mock_get):
        """Test workflow with fallback to alternative sources."""
        metadata_response = Mock()
        metadata_response.json.return_value = {
            "records": [
                {"pmid": "12345678", "pmcid": "PMC123456", "doi": "10.1234/test"}
            ]
        }
        metadata_response.raise_for_status = Mock()

        oa_response = Mock()
        oa_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <OA><error code="idDoesNotExist">Invalid PMC ID</error></OA>"""
        oa_response.raise_for_status = Mock()

        scrape_response = Mock()
        html = (
            '<html><head><meta name="citation_pdf_url" '
            'content="https://www.ncbi.nlm.nih.gov/pmc/articles/'
            'PMC123456/pdf/fallback.pdf"></head></html>'
        )
        scrape_response.content = html.encode()
        scrape_response.raise_for_status = Mock()

        mock_get.side_effect = [metadata_response, oa_response, scrape_response]

        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 404
            identifier = "12345678"
            metadata = self.downloader.fetch_metadata(identifier)
            pdf_url = self.downloader.construct_pdf_url(metadata, identifier)

        self.assertEqual(
            pdf_url,
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/fallback.pdf",
        )
        self.assertEqual(mock_get.call_count, 3)
        mock_head.assert_called_once()
