"""paper_loader tests for the load_all_papers function."""

import pytest
from unittest.mock import patch, MagicMock

from aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader import (
    load_all_papers,
)


@pytest.fixture
def articles():
    """a fixture to provide a sample articles dictionary."""
    return {
        "p1": {"pdf_url": "http://example.com/p1.pdf", "title": "Paper 1"},
        "p2": {"pdf_url": "http://example.com/p2.pdf", "title": "Paper 2"},
        "p3": {"title": "No PDF paper"},
    }


@pytest.fixture
def mock_vector_store():
    """mock vector store fixture."""
    return MagicMock(
        loaded_papers={"p1"},
        paper_metadata={},
        documents={},
        metadata_fields=["title"],
        config={"embedding_batch_size": 1234},
        has_gpu=False,
        vector_store=MagicMock(),
    )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_all_papers_loaded_returns_early(mock_batch, articles, mock_vector_store):
    """all_papers_loaded should return early if all papers are already loaded."""
    # Mark all as already loaded
    mock_vector_store.loaded_papers = set(articles.keys())

    load_all_papers(
        vector_store=mock_vector_store,
        articles=articles,
        call_id="test_call",
        config={"embedding_batch_size": 1000},
        has_gpu=False,
    )

    mock_batch.assert_not_called()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_skips_papers_without_pdf(mock_batch, articles, mock_vector_store):
    """should skip papers that do not have a PDF URL."""
    mock_vector_store.loaded_papers = {"p2"}  # p1 not loaded, p3 has no pdf

    load_all_papers(
        vector_store=mock_vector_store,
        articles=articles,
        call_id="test_call",
        config={"embedding_batch_size": 1000},
        has_gpu=False,
    )

    # Should call add_papers_batch only for p1
    assert mock_batch.call_count == 1
    call_args = mock_batch.call_args[1]["papers_to_add"]
    assert len(call_args) == 1
    assert call_args[0][0] == "p1"


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_gpu_parameters_used(mock_batch, articles, mock_vector_store):
    """gpu parameters should be used when has_gpu is True."""
    mock_vector_store.loaded_papers = set()
    mock_vector_store.has_gpu = True

    load_all_papers(
        vector_store=mock_vector_store,
        articles=articles,
        call_id="gpu_call",
        config={"embedding_batch_size": 2048},
        has_gpu=True,
    )

    args = mock_batch.call_args[1]
    assert args["has_gpu"] is True
    assert args["batch_size"] == 2048
    assert args["max_workers"] >= 4


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
def test_cpu_parameters_used(mock_batch, articles, mock_vector_store):
    """cpu parameters should be used when has_gpu is False."""
    mock_vector_store.loaded_papers = set()
    mock_vector_store.has_gpu = False

    load_all_papers(
        vector_store=mock_vector_store,
        articles=articles,
        call_id="cpu_call",
        config={"embedding_batch_size": 512},
        has_gpu=False,
    )

    args = mock_batch.call_args[1]
    assert args["has_gpu"] is False
    assert args["batch_size"] == 512
    assert args["max_workers"] >= 3


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.add_papers_batch")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader.logger")
def test_exception_in_batch_loader(
    mock_logger, mock_batch, articles, mock_vector_store
):
    """exception in batch loading should be logged."""
    mock_vector_store.loaded_papers = set()
    mock_batch.side_effect = RuntimeError("batch failed")

    load_all_papers(
        vector_store=mock_vector_store,
        articles=articles,
        call_id="error_call",
        config={"embedding_batch_size": 1000},
        has_gpu=False,
    )

    mock_logger.error.assert_called()
    error_msg = mock_logger.error.call_args[0][0]
    assert "Error during batch paper loading" in error_msg
