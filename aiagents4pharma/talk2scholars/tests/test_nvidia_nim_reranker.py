"""
Unit tests for NVIDIA NIM reranker error handling in nvidia_nim_reranker.py
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker import (
    rerank_chunks,
)


@pytest.fixture
def mock_chunks():
    return [
        Document(
            page_content=f"chunk {i}",
            metadata={"paper_id": f"P{i%2}", "relevance_score": 0.9 - 0.01 * i},
        )
        for i in range(10)
    ]


def test_rerank_chunks_short_input(mock_chunks):
    result = rerank_chunks(
        mock_chunks[:3], "What is cancer?", config=MagicMock(), top_k=5
    )
    assert result == mock_chunks[:3]  # Should return original since len <= top_k


def test_rerank_chunks_missing_api_key(mock_chunks):
    mock_config = MagicMock()
    mock_config.reranker.api_key = None

    result = rerank_chunks(mock_chunks, "What is cancer?", config=mock_config, top_k=5)
    assert result == mock_chunks[:5]  # fallback triggered


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.NVIDIARerank")
def test_rerank_chunks_success(mock_reranker_cls, mock_chunks):
    # Fake reranker returns reversed list
    reranker_instance = MagicMock()
    reranker_instance.compress_documents.return_value = list(reversed(mock_chunks))
    mock_reranker_cls.return_value = reranker_instance

    mock_config = MagicMock()
    mock_config.reranker.api_key = "test_key"
    mock_config.reranker.model = "test_model"

    result = rerank_chunks(
        mock_chunks, "Explain mitochondria.", config=mock_config, top_k=5
    )

    assert isinstance(result, list)
    assert result == list(reversed(mock_chunks))[:5]
    reranker_instance.compress_documents.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.NVIDIARerank")
def test_rerank_chunks_reranker_fails(mock_reranker_cls, mock_chunks):
    reranker_instance = MagicMock()
    reranker_instance.compress_documents.side_effect = RuntimeError("API failure")
    mock_reranker_cls.return_value = reranker_instance

    mock_config = MagicMock()
    mock_config.reranker.api_key = "valid_key"
    mock_config.reranker.model = "reranker"

    result = rerank_chunks(
        mock_chunks, "How does light affect plants?", config=mock_config, top_k=3
    )

    # Should fallback to first 3
    assert result == mock_chunks[:3]


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.logger")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.NVIDIARerank")
def test_rerank_chunks_debug_block_triggered(mock_reranker_cls, mock_logger):
    # Force logger.isEnabledFor(logging.DEBUG) → True
    mock_logger.isEnabledFor.return_value = True

    # Simulate reranker returns chunks with paper_ids
    chunks = [
        Document(page_content="a", metadata={"paper_id": "P1"}),
        Document(page_content="b", metadata={"paper_id": "P2"}),
        Document(page_content="c", metadata={"paper_id": "P1"}),
    ]

    reranker_instance = MagicMock()
    reranker_instance.compress_documents.return_value = chunks
    mock_reranker_cls.return_value = reranker_instance

    mock_config = MagicMock()
    mock_config.reranker.api_key = "abc"
    mock_config.reranker.model = "mymodel"

    from aiagents4pharma.talk2scholars.tools.pdf.utils import nvidia_nim_reranker

    result = nvidia_nim_reranker.rerank_chunks(
        chunks * 2, "Test query", mock_config, top_k=3
    )

    assert result == chunks[:3]
    assert mock_logger.debug.called
