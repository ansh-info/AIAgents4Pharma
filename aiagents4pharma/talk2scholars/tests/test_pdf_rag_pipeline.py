"""pdf rag pipeline tests."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline import (
    retrieve_and_rerank_chunks,
)


@pytest.fixture
def base_config():
    """base_config fixture to provide common configuration for tests."""

    class Config:
        """configuration class for tests."""

        def get(self, key, default=None):
            """get method to retrieve configuration values."""
            return {
                "initial_retrieval_k": 120,
                "mmr_diversity": 0.7,
            }.get(key, default)

        top_k_chunks = 5

    return Config()


@pytest.fixture
def mock_docs():
    """mock_docs fixture to simulate PDF chunks."""
    return [
        Document(page_content=f"chunk {i}", metadata={"paper_id": f"P{i % 2}"})
        for i in range(10)
    ]


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.rerank_chunks")
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.retrieve_relevant_chunks"
)
def test_rag_pipeline_gpu_path(mock_retrieve, mock_rerank, base_config, mock_docs):
    """test RAG pipeline with GPU path."""
    mock_retrieve.return_value = mock_docs
    mock_rerank.return_value = mock_docs[:5]

    result = retrieve_and_rerank_chunks(
        vector_store=MagicMock(),
        query="Explain AI.",
        config=base_config,
        call_id="gpu_test",
        has_gpu=True,
    )

    assert result == mock_docs[:5]
    mock_retrieve.assert_called_once()
    mock_rerank.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.rerank_chunks")
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.retrieve_relevant_chunks"
)
def test_rag_pipeline_cpu_path(mock_retrieve, mock_rerank, base_config, mock_docs):
    """rag pipeline with CPU path."""
    mock_retrieve.return_value = mock_docs
    mock_rerank.return_value = mock_docs[:5]

    result = retrieve_and_rerank_chunks(
        vector_store=MagicMock(),
        query="Explain quantum physics.",
        config=base_config,
        call_id="cpu_test",
        has_gpu=False,
    )

    assert result == mock_docs[:5]
    mock_retrieve.assert_called_once()
    mock_rerank.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.rerank_chunks")
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.rag_pipeline.retrieve_relevant_chunks"
)
def test_rag_pipeline_empty_results(mock_retrieve, mock_rerank, base_config):
    """rag pipeline with no results."""
    mock_retrieve.return_value = []

    result = retrieve_and_rerank_chunks(
        vector_store=MagicMock(),
        query="No match?",
        config=base_config,
        call_id="empty_test",
        has_gpu=False,
    )

    assert result == []
    mock_rerank.assert_not_called()
