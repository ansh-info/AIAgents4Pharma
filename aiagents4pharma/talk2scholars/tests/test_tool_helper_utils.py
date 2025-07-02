"""
Unit tests for QAToolHelper routines in tool_helper.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper import QAToolHelper


class TestQAToolHelper(unittest.TestCase):
    """tests for QAToolHelper routines in tool_helper.py"""

    def setUp(self):
        """set up test case"""
        self.helper = QAToolHelper()

    def test_start_call_sets_config_and_call_id(self):
        """test start_call sets config and call_id"""
        cfg = SimpleNamespace(foo="bar")
        self.helper.start_call(cfg, "call123")
        self.assertIs(self.helper.config, cfg)
        self.assertEqual(self.helper.call_id, "call123")

    def test_init_vector_store_reuse(self):
        """test init_vector_store reuses existing instance"""
        emb_model = MagicMock()
        first = self.helper.init_vector_store(emb_model)
        second = self.helper.init_vector_store(emb_model)
        self.assertIs(second, first)

    def test_get_state_models_and_data_success(self):
        """test get_state_models_and_data returns models and data"""
        emb = MagicMock()
        llm = MagicMock()
        articles = {"p": {}}
        state = {
            "text_embedding_model": emb,
            "llm_model": llm,
            "article_data": articles,
        }
        ret_emb, ret_llm, ret_articles = self.helper.get_state_models_and_data(state)
        self.assertIs(ret_emb, emb)
        self.assertIs(ret_llm, llm)
        self.assertIs(ret_articles, articles)

    def test_get_state_models_and_data_missing_text_embedding(self):
        """test get_state_models_and_data raises ValueError if missing text embedding"""
        state = {"llm_model": MagicMock(), "article_data": {"p": {}}}
        with self.assertRaises(ValueError) as cm:
            self.helper.get_state_models_and_data(state)
        self.assertEqual(str(cm.exception), "No text embedding model found in state.")

    def test_get_state_models_and_data_missing_llm(self):
        """test get_state_models_and_data raises ValueError if missing LLM"""
        state = {"text_embedding_model": MagicMock(), "article_data": {"p": {}}}
        with self.assertRaises(ValueError) as cm:
            self.helper.get_state_models_and_data(state)
        self.assertEqual(str(cm.exception), "No LLM model found in state.")

    def test_get_state_models_and_data_missing_article_data(self):
        """test get_state_models_and_data raises ValueError if missing article data"""
        state = {"text_embedding_model": MagicMock(), "llm_model": MagicMock()}
        with self.assertRaises(ValueError) as cm:
            self.helper.get_state_models_and_data(state)
        self.assertEqual(str(cm.exception), "No article_data found in state.")

    def test_get_hardware_stats(self):
        helper = QAToolHelper()
        helper.call_id = "test_call"

        # Case 1: GPU not available
        helper.has_gpu = False
        stats = helper.get_hardware_stats()
        assert stats == {
            "gpu_available": False,
            "hardware_mode": "CPU-only",
            "call_id": "test_call",
        }

        # Case 2: GPU available
        helper.has_gpu = True
        stats = helper.get_hardware_stats()
        assert stats == {
            "gpu_available": True,
            "hardware_mode": "GPU-accelerated",
            "call_id": "test_call",
        }
