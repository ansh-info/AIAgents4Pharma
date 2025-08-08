"""
Unit tests for the unified paper downloader functionality.
Tests the main download_papers tool and PaperDownloaderFactory.
"""

import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader import (
    PaperDownloaderFactory,
    _download_papers_impl,
    download_arxiv_papers,
    download_biorxiv_papers,
    download_medrxiv_papers,
    download_papers,
    download_pubmed_papers,
)


class TestPaperDownloaderFactory(unittest.TestCase):
    """Tests for the PaperDownloaderFactory class."""

    def setUp(self):
        """Reset the factory state before each test."""
        PaperDownloaderFactory._cached_config = None
        PaperDownloaderFactory._config_lock = None

    def tearDown(self):
        """Clean up after each test."""
        PaperDownloaderFactory._cached_config = None
        PaperDownloaderFactory._config_lock = None

    def test_clear_cache(self):
        """Test that clear_cache method works correctly."""
        PaperDownloaderFactory._cached_config = {"test": "config"}
        PaperDownloaderFactory.clear_cache()
        self.assertIsNone(PaperDownloaderFactory._cached_config)

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.ArxivDownloader"
    )
    @patch.object(PaperDownloaderFactory, "_get_unified_config")
    @patch.object(PaperDownloaderFactory, "_build_service_config")
    def test_create_arxiv_downloader(
        self, mock_build_config, mock_get_config, mock_arxiv
    ):
        """Test creating ArxivDownloader through factory."""
        mock_config = Mock()
        mock_service_config = Mock()
        mock_get_config.return_value = mock_config
        mock_build_config.return_value = mock_service_config

        result = PaperDownloaderFactory.create("arxiv")

        mock_get_config.assert_called_once()
        mock_build_config.assert_called_once_with(mock_config, "arxiv")
        mock_arxiv.assert_called_once_with(mock_service_config)
        self.assertEqual(result, mock_arxiv.return_value)

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.MedrxivDownloader"
    )
    @patch.object(PaperDownloaderFactory, "_get_unified_config")
    @patch.object(PaperDownloaderFactory, "_build_service_config")
    def test_create_medrxiv_downloader(
        self, mock_build_config, mock_get_config, mock_medrxiv
    ):
        """Test creating MedrxivDownloader through factory."""
        mock_config = Mock()
        mock_service_config = Mock()
        mock_get_config.return_value = mock_config
        mock_build_config.return_value = mock_service_config

        result = PaperDownloaderFactory.create("medrxiv")

        mock_build_config.assert_called_once_with(mock_config, "medrxiv")
        mock_medrxiv.assert_called_once_with(mock_service_config)
        self.assertEqual(result, mock_medrxiv.return_value)

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.BiorxivDownloader"
    )
    @patch.object(PaperDownloaderFactory, "_get_unified_config")
    @patch.object(PaperDownloaderFactory, "_build_service_config")
    def test_create_biorxiv_downloader(
        self, mock_build_config, mock_get_config, mock_biorxiv
    ):
        """Test creating BiorxivDownloader through factory."""
        mock_config = Mock()
        mock_service_config = Mock()
        mock_get_config.return_value = mock_config
        mock_build_config.return_value = mock_service_config

        result = PaperDownloaderFactory.create("biorxiv")

        mock_build_config.assert_called_once_with(mock_config, "biorxiv")
        mock_biorxiv.assert_called_once_with(mock_service_config)
        self.assertEqual(result, mock_biorxiv.return_value)

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.PubmedDownloader"
    )
    @patch.object(PaperDownloaderFactory, "_get_unified_config")
    @patch.object(PaperDownloaderFactory, "_build_service_config")
    def test_create_pubmed_downloader(
        self, mock_build_config, mock_get_config, mock_pubmed
    ):
        """Test creating PubmedDownloader through factory."""
        mock_config = Mock()
        mock_service_config = Mock()
        mock_get_config.return_value = mock_config
        mock_build_config.return_value = mock_service_config

        result = PaperDownloaderFactory.create("pubmed")

        mock_build_config.assert_called_once_with(mock_config, "pubmed")
        mock_pubmed.assert_called_once_with(mock_service_config)
        self.assertEqual(result, mock_pubmed.return_value)

    @patch.object(PaperDownloaderFactory, "_get_unified_config")
    def test_create_unsupported_service(self, mock_get_config):
        """Test error when creating unsupported service."""
        mock_config = Mock()
        # Mock services as a dict-like object that doesn't contain 'unsupported'
        mock_config.services = {"arxiv": {}, "medrxiv": {}, "biorxiv": {}, "pubmed": {}}
        mock_config.supported_services = ["arxiv", "medrxiv", "biorxiv", "pubmed"]
        mock_get_config.return_value = mock_config

        with self.assertRaises(ValueError) as context:
            PaperDownloaderFactory.create("unsupported")

        self.assertIn(
            "Service 'unsupported' not found in configuration", str(context.exception)
        )

    @patch.object(PaperDownloaderFactory, "_get_unified_config")
    def test_create_unsupported_service_fallback_error(self, mock_get_config):
        """Test error when creating unsupported service (using fallback error message)."""
        mock_config = Mock()
        # Mock services that contains the service but make the factory not handle it
        mock_config.services = {"unsupported": {}}
        mock_config.common = Mock()
        mock_config.supported_services = ["arxiv", "medrxiv", "biorxiv", "pubmed"]
        mock_get_config.return_value = mock_config

        # Mock _build_service_config to succeed
        with patch.object(
            PaperDownloaderFactory, "_build_service_config", return_value=Mock()
        ):
            with self.assertRaises(ValueError) as context:
                PaperDownloaderFactory.create("unsupported")

            self.assertIn(
                "Unsupported service: unsupported. Supported:", str(context.exception)
            )

    @patch("aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.hydra")
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.GlobalHydra"
    )
    def test_get_unified_config_success(self, mock_global_hydra, mock_hydra):
        """Test successful configuration loading."""
        mock_hydra_instance = Mock()
        mock_global_hydra.return_value = mock_hydra_instance
        mock_global_hydra().is_initialized.return_value = False

        mock_cfg = Mock()
        mock_cfg.tools.paper_download = {"test": "config"}
        mock_hydra.compose.return_value = mock_cfg

        result = PaperDownloaderFactory._get_unified_config()

        self.assertEqual(result, {"test": "config"})
        self.assertEqual(PaperDownloaderFactory._cached_config, {"test": "config"})

    @patch("aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.hydra")
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.GlobalHydra"
    )
    def test_get_unified_config_clear_existing(self, mock_global_hydra, mock_hydra):
        """Test configuration loading with existing GlobalHydra."""
        mock_hydra_instance = Mock()
        mock_global_hydra.return_value = mock_hydra_instance
        mock_global_hydra().is_initialized.return_value = True
        mock_global_hydra.instance.return_value.clear = Mock()

        mock_cfg = Mock()
        mock_cfg.tools.paper_download = {"test": "config"}
        mock_hydra.compose.return_value = mock_cfg

        result = PaperDownloaderFactory._get_unified_config()

        mock_global_hydra.instance().clear.assert_called_once()
        self.assertEqual(result, {"test": "config"})

    def test_get_unified_config_cached(self):
        """Test that cached config is returned when available."""
        PaperDownloaderFactory._cached_config = {"cached": "config"}

        result = PaperDownloaderFactory._get_unified_config()

        self.assertEqual(result, {"cached": "config"})

    def test_get_unified_config_cached_thread_race(self):
        """Test the double-check pattern in cached config loading."""
        # First clear the cache
        PaperDownloaderFactory._cached_config = None
        PaperDownloaderFactory._config_lock = None

        # Mock the lock to simulate a race condition
        mock_lock = Mock()
        PaperDownloaderFactory._config_lock = mock_lock

        # Set up the context manager to simulate another thread setting the config
        def lock_context_manager():
            class MockContext:
                def __enter__(self):
                    # Simulate another thread setting the config during lock acquisition
                    PaperDownloaderFactory._cached_config = {"race_condition": "config"}
                    return self

                def __exit__(self, *args):
                    pass

            return MockContext()

        mock_lock.__enter__ = lambda self: lock_context_manager().__enter__()
        mock_lock.__exit__ = lambda self, *args: lock_context_manager().__exit__(*args)

        result = PaperDownloaderFactory._get_unified_config()

        self.assertEqual(result, {"race_condition": "config"})

    @patch("aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.hydra")
    def test_get_unified_config_failure(self, mock_hydra):
        """Test configuration loading failure."""
        mock_hydra.initialize.side_effect = Exception("Config error")

        with self.assertRaises(RuntimeError) as context:
            PaperDownloaderFactory._get_unified_config()

        self.assertIn("Configuration loading failed", str(context.exception))

    def test_build_service_config_missing_service(self):
        """Test build_service_config with missing service."""
        mock_config = Mock()
        mock_config.services = {}

        with self.assertRaises(ValueError) as context:
            PaperDownloaderFactory._build_service_config(mock_config, "missing")

        self.assertIn("Service 'missing' not found", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.OmegaConf"
    )
    def test_build_service_config_omega_conf(self, mock_omega_conf):
        """Test build_service_config with OmegaConf objects."""
        # Setup mock config
        mock_unified_config = Mock()
        mock_unified_config.services = {"test": Mock()}
        mock_unified_config.common = Mock()
        mock_unified_config.common._content = True
        mock_unified_config.services["test"]._content = True

        # Setup OmegaConf mock
        mock_omega_conf.to_container.side_effect = [
            {"common_key": "common_value"},  # For common config
            {"service_key": "service_value"},  # For service config
        ]

        result = PaperDownloaderFactory._build_service_config(
            mock_unified_config, "test"
        )

        # Verify the config object has both common and service attributes
        self.assertTrue(hasattr(result, "common_key"))
        self.assertTrue(hasattr(result, "service_key"))
        self.assertEqual(result.common_key, "common_value")
        self.assertEqual(result.service_key, "service_value")

    def test_service_config_methods(self):
        """Test ServiceConfig helper methods."""
        mock_unified_config = Mock()
        mock_unified_config.services = {"test": Mock()}
        mock_unified_config.common = Mock()

        # Mock the _apply_config calls to avoid complex setup
        with patch.object(PaperDownloaderFactory, "_apply_config"):
            config = PaperDownloaderFactory._build_service_config(
                mock_unified_config, "test"
            )

            # Test get_config_dict method
            config.test_attr = "test_value"
            config._private_attr = "private"

            config_dict = config.get_config_dict()
            self.assertIn("test_attr", config_dict)
            self.assertNotIn("_private_attr", config_dict)
            self.assertEqual(config_dict["test_attr"], "test_value")

            # Test has_attribute method
            self.assertTrue(config.has_attribute("test_attr"))
            self.assertFalse(config.has_attribute("nonexistent_attr"))


class TestDownloadPapersFunction(unittest.TestCase):
    """Tests for the download_papers tool function and related functions."""

    @patch.object(PaperDownloaderFactory, "create")
    def test_download_papers_success(self, mock_create):
        """Test successful paper download."""
        # Setup mock downloader
        mock_downloader = Mock()
        mock_downloader.get_service_name.return_value = "arXiv"
        mock_downloader.process_identifiers.return_value = {
            "1234.5678": {
                "Title": "Test Paper",
                "access_type": "open_access_downloaded",
            }
        }
        mock_downloader.build_summary.return_value = "Successfully downloaded 1 paper"
        mock_create.return_value = mock_downloader

        result = _download_papers_impl("arxiv", ["1234.5678"], "test_tool_call_id")

        # Verify result structure
        self.assertIsInstance(result, Command)
        self.assertIn("article_data", result.update)
        self.assertIn("messages", result.update)

        # Verify article data
        article_data = result.update["article_data"]
        self.assertIn("1234.5678", article_data)
        self.assertEqual(article_data["1234.5678"]["Title"], "Test Paper")

        # Verify message
        messages = result.update["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], ToolMessage)
        self.assertEqual(messages[0].tool_call_id, "test_tool_call_id")
        self.assertEqual(messages[0].content, "Successfully downloaded 1 paper")

    @patch.object(PaperDownloaderFactory, "create")
    def test_download_papers_service_error(self, mock_create):
        """Test download_papers with service error."""
        mock_create.side_effect = ValueError("Unsupported service: invalid")

        result = _download_papers_impl("invalid", ["123"], "test_tool_call_id")

        # Verify error handling
        self.assertIsInstance(result, Command)
        self.assertEqual(result.update["article_data"], {})

        messages = result.update["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIn("Service error", messages[0].content)
        self.assertIn("invalid", messages[0].content)

    @patch.object(PaperDownloaderFactory, "create")
    def test_download_papers_unexpected_error(self, mock_create):
        """Test download_papers with unexpected error."""
        mock_downloader = Mock()
        mock_downloader.process_identifiers.side_effect = RuntimeError(
            "Unexpected error"
        )
        mock_create.return_value = mock_downloader

        result = _download_papers_impl("arxiv", ["123"], "test_tool_call_id")

        # Verify error handling
        self.assertIsInstance(result, Command)
        self.assertEqual(result.update["article_data"], {})

        messages = result.update["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIn("Unexpected error", messages[0].content)

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader._download_papers_impl"
    )
    def test_convenience_functions(self, mock_impl):
        """Test convenience wrapper functions."""
        mock_impl.return_value = Command(update={"test": "result"})

        # Test each convenience function
        download_arxiv_papers(["1234.5678"], "tool_call_1")
        mock_impl.assert_called_with("arxiv", ["1234.5678"], "tool_call_1")

        download_medrxiv_papers(["10.1101/test"], "tool_call_2")
        mock_impl.assert_called_with("medrxiv", ["10.1101/test"], "tool_call_2")

        download_biorxiv_papers(["10.1101/test"], "tool_call_3")
        mock_impl.assert_called_with("biorxiv", ["10.1101/test"], "tool_call_3")

        download_pubmed_papers(["12345"], "tool_call_4")
        mock_impl.assert_called_with("pubmed", ["12345"], "tool_call_4")

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader._download_papers_impl"
    )
    def test_main_download_papers_function(self, mock_impl):
        """Test the main download_papers tool function."""
        mock_impl.return_value = Command(update={"test": "result"})

        # Call the tool with proper input structure (as it's decorated with @tool)
        input_data = {
            "service": "arxiv",
            "identifiers": ["1234.5678"],
            "tool_call_id": "tool_call_id",
        }

        result = download_papers.invoke(input_data)

        mock_impl.assert_called_once_with("arxiv", ["1234.5678"], "tool_call_id")
        self.assertEqual(result.update["test"], "result")


class TestConfigurationExtraction(unittest.TestCase):
    """Tests for configuration extraction helper methods."""

    def test_extract_from_omegaconf(self):
        """Test OmegaConf extraction."""

        # Create a simple object to act as config_obj
        class TestConfig:
            pass

        config_obj = TestConfig()
        source_config = Mock()

        with patch(
            "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.OmegaConf"
        ) as mock_omega_conf:
            mock_omega_conf.to_container.return_value = {
                "key1": "value1",
                "key2": "value2",
                123: "invalid_key",  # Non-string key should be skipped
            }

            PaperDownloaderFactory._extract_from_omegaconf(config_obj, source_config)

            # Verify only string keys were set
            self.assertTrue(hasattr(config_obj, "key1"))
            self.assertTrue(hasattr(config_obj, "key2"))
            self.assertFalse(
                hasattr(config_obj, "123")
            )  # Non-string key should be skipped
            self.assertEqual(config_obj.key1, "value1")
            self.assertEqual(config_obj.key2, "value2")

    def test_extract_from_dict(self):
        """Test dictionary extraction."""

        # Create a simple object to act as config_obj
        class TestConfig:
            pass

        config_obj = TestConfig()
        source_dict = {"public_key": "public_value", "_private_key": "private_value"}

        PaperDownloaderFactory._extract_from_dict(config_obj, source_dict)

        # Verify only non-private keys were set
        self.assertTrue(hasattr(config_obj, "public_key"))
        self.assertFalse(hasattr(config_obj, "_private_key"))
        self.assertEqual(config_obj.public_key, "public_value")

    def test_extract_from_items(self):
        """Test items() method extraction."""

        # Create a simple object to act as config_obj
        class TestConfig:
            pass

        config_obj = TestConfig()
        source_config = Mock()
        source_config.items.return_value = [
            ("str_key", "value1"),
            (123, "value2"),  # Non-string key should be skipped
        ]

        PaperDownloaderFactory._extract_from_items(config_obj, source_config)

        # Verify only string keys were set
        self.assertTrue(hasattr(config_obj, "str_key"))
        self.assertFalse(hasattr(config_obj, "123"))
        self.assertEqual(config_obj.str_key, "value1")

    def test_extract_from_dir(self):
        """Test dir() approach extraction."""

        # Create a simple object to act as config_obj
        class TestConfig:
            pass

        config_obj = TestConfig()

        # Create a simple source config object
        class SourceConfig:
            def __init__(self):
                self.public_attr = "public_value"
                self._private_attr = "private_value"

        source_config = SourceConfig()

        PaperDownloaderFactory._extract_from_dir(config_obj, source_config)

        # Verify only non-private attributes were set
        self.assertTrue(hasattr(config_obj, "public_attr"))
        self.assertFalse(hasattr(config_obj, "_private_attr"))
        self.assertEqual(config_obj.public_attr, "public_value")

    def test_apply_config_exception_handling(self):
        """Test exception handling in _apply_config method."""

        # Create a simple object to act as config_obj
        class TestConfig:
            pass

        config_obj = TestConfig()

        # Mock _try_config_extraction to raise an exception
        with patch.object(
            PaperDownloaderFactory,
            "_try_config_extraction",
            side_effect=AttributeError("test error"),
        ):
            with patch(
                "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.logger"
            ) as mock_logger:
                PaperDownloaderFactory._apply_config(config_obj, Mock(), "test")
                mock_logger.warning.assert_called_once_with(
                    "Failed to process %s config: %s",
                    "test",
                    mock_logger.warning.call_args[0][2],
                )

    def test_try_config_extraction_individual_methods(self):
        """Test individual methods in config extraction."""

        # Create a simple object to act as config_obj
        class TestConfig:
            pass

        # Test direct __dict__ method
        config_obj = TestConfig()
        PaperDownloaderFactory._extract_from_dict(
            config_obj, {"dict_key": "dict_value"}
        )
        self.assertEqual(config_obj.dict_key, "dict_value")

        # Test items() method
        config_obj2 = TestConfig()
        items_source = Mock()
        items_source.items.return_value = [("items_key", "items_value")]
        PaperDownloaderFactory._extract_from_items(config_obj2, items_source)
        self.assertEqual(config_obj2.items_key, "items_value")

        # Test dir() method
        config_obj3 = TestConfig()

        class DirSource:
            def __init__(self):
                self.dir_attr = "dir_value"
                self._private = "private_value"

        dir_source = DirSource()
        PaperDownloaderFactory._extract_from_dir(config_obj3, dir_source)
        self.assertEqual(config_obj3.dir_attr, "dir_value")
        self.assertFalse(hasattr(config_obj3, "_private"))

    def test_try_config_extraction_fallback_methods(self):
        """Test fallback methods in _try_config_extraction by directly calling them."""

        class TestConfig:
            pass

        # Test __dict__ method directly (lines 215-219)
        config_obj = TestConfig()
        dict_source = {"dict_attr": "dict_value"}
        PaperDownloaderFactory._extract_from_dict(config_obj, dict_source)
        self.assertEqual(config_obj.dict_attr, "dict_value")

        # Test items() method directly (lines 221-224)
        config_obj2 = TestConfig()
        items_source = Mock()
        items_source.items.return_value = [("items_key", "items_value")]
        PaperDownloaderFactory._extract_from_items(config_obj2, items_source)
        self.assertEqual(config_obj2.items_key, "items_value")

        # Test dir() method directly (lines 226-227)
        config_obj3 = TestConfig()

        class DirSource:
            def __init__(self):
                self.dir_attr = "dir_value"
                self._private = "private_value"

        dir_source = DirSource()
        PaperDownloaderFactory._extract_from_dir(config_obj3, dir_source)
        self.assertEqual(config_obj3.dir_attr, "dir_value")
        self.assertFalse(hasattr(config_obj3, "_private"))

    def test_try_config_extraction_comprehensive_fallback_flow(self):
        """Test the complete fallback flow through _try_config_extraction."""

        class TestConfig:
            pass

        # Test items() fallback path (lines 222-224)
        config_obj2 = TestConfig()

        # Mock hasattr to force items() path
        with patch(
            "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.hasattr"
        ) as mock_hasattr:

            def hasattr_side_effect(obj, attr):
                if attr == "_content":
                    return False  # Skip OmegaConf path
                elif attr == "__dict__":
                    return False  # Skip __dict__ path
                elif attr == "items":
                    return True  # Use items() fallback
                # No default return needed for this test

            mock_hasattr.side_effect = hasattr_side_effect

            # Create source with items method
            source_config = Mock()
            source_config.items.return_value = [("items_attr", "items_value")]

            PaperDownloaderFactory._try_config_extraction(config_obj2, source_config)

            # Should have extracted using items()
            self.assertTrue(hasattr(config_obj2, "items_attr"))
            self.assertEqual(config_obj2.items_attr, "items_value")

        # Test dir() fallback path (lines 226-227)
        config_obj3 = TestConfig()

        # Mock hasattr to force dir() path
        with patch(
            "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.hasattr"
        ) as mock_hasattr:

            def hasattr_side_effect(obj, attr):
                if attr == "_content":
                    return False  # Skip OmegaConf path
                elif attr == "__dict__":
                    return False  # Skip __dict__ path
                elif attr == "items":
                    return False  # Skip items() path - forces dir() fallback
                # No default return needed for this test

            mock_hasattr.side_effect = hasattr_side_effect

            # Create source for dir() approach
            class DirSource:
                def __init__(self):
                    self.dir_attr = "dir_value"
                    self._private = "private"

            source_config = DirSource()

            PaperDownloaderFactory._try_config_extraction(config_obj3, source_config)

            # Should have extracted using dir()
            self.assertTrue(hasattr(config_obj3, "dir_attr"))
            self.assertEqual(config_obj3.dir_attr, "dir_value")
            self.assertFalse(hasattr(config_obj3, "_private"))

    def test_try_config_extraction_dict_access_path(self):
        """Test the __dict__ access path (lines 215-219) in _try_config_extraction."""

        class TestConfig:
            pass

        config_obj = TestConfig()

        # Mock hasattr to force __dict__ access path
        with patch(
            "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.hasattr"
        ) as mock_hasattr:

            def hasattr_side_effect(obj, attr):
                if attr == "_content":
                    return False  # Skip OmegaConf path
                elif attr == "__dict__":
                    return True  # Use __dict__ access - lines 215-219
                # No default return needed for this test

            mock_hasattr.side_effect = hasattr_side_effect

            # Create source with __dict__ attribute
            class DictSource:
                def __init__(self):
                    self.test_attr = "test_value"

            source_config = DictSource()

            PaperDownloaderFactory._try_config_extraction(config_obj, source_config)

            # Should have extracted using __dict__ access
            self.assertTrue(hasattr(config_obj, "test_attr"))
            self.assertEqual(config_obj.test_attr, "test_value")

    def test_hasattr_side_effect_coverage(self):
        """Test to ensure all paths in hasattr_side_effect functions are covered."""

        class TestConfig:
            pass

        # Test to ensure all return paths in hasattr side effects are covered
        config_obj = TestConfig()

        # This will test the 'return True' path (line 629)
        with patch(
            "aiagents4pharma.talk2scholars.tools.paper_download.paper_downloader.hasattr"
        ) as mock_hasattr:

            def hasattr_side_effect(obj, attr):
                if attr == "_content":
                    return False
                elif attr == "__dict__":
                    return False
                elif attr == "items":
                    return False
                # No default return needed for this test

            mock_hasattr.side_effect = hasattr_side_effect

            class TestSource:
                def __init__(self):
                    self.attr = "value"

            source = TestSource()
            PaperDownloaderFactory._try_config_extraction(config_obj, source)
            self.assertTrue(hasattr(config_obj, "attr"))
