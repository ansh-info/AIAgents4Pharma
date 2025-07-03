"""retrieve_chunks for PDF tool tests"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks import (
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_with_scores,
)


@pytest.fixture
def mock_vector_store():
    """vector_store fixture to simulate a vector store."""
    return MagicMock()


@pytest.fixture
def mock_chunks():
    """mock_chunks fixture to simulate PDF chunks."""
    return [
        Document(page_content=f"chunk {i}", metadata={"paper_id": f"P{i%2}"})
        for i in range(5)
    ]


@pytest.fixture
def mock_scored_chunks():
    """mock_scored_chunks fixture to simulate scored PDF chunks."""
    return [
        (Document(page_content=f"chunk {i}", metadata={}), score)
        for i, score in enumerate([0.9, 0.8, 0.4, 0.95])
    ]


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_cpu_success(mock_logger, mock_vector_store, mock_chunks):
    """ "test retrieve_relevant_chunks with CPU path."""
    mock_vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    mock_vector_store.max_marginal_relevance_search.return_value = mock_chunks

    results = retrieve_relevant_chunks(mock_vector_store, query="AI", top_k=5)

    assert results == mock_chunks
    mock_vector_store.max_marginal_relevance_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_gpu_success(mock_logger, mock_vector_store, mock_chunks):
    """test retrieve_relevant_chunks with GPU path."""
    mock_vector_store.has_gpu = True
    mock_logger.debug = MagicMock()
    mock_vector_store.max_marginal_relevance_search.return_value = mock_chunks

    results = retrieve_relevant_chunks(mock_vector_store, query="AI", top_k=5)

    assert results == mock_chunks
    mock_vector_store.max_marginal_relevance_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_fallback_on_exception(
    mock_logger, mock_vector_store, mock_chunks
):
    """test retrieve_relevant_chunks fallback to similarity search on exception."""
    mock_vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    mock_vector_store.max_marginal_relevance_search.side_effect = Exception("MMR fail")
    mock_vector_store.similarity_search.return_value = mock_chunks

    results = retrieve_relevant_chunks(
        mock_vector_store, query="Fallback test", top_k=3
    )
    assert results == mock_chunks
    mock_vector_store.similarity_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_final_failure_returns_empty(mock_logger, mock_vector_store):
    """test retrieve_relevant_chunks returns empty on final failure."""
    mock_vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    mock_vector_store.max_marginal_relevance_search.side_effect = Exception("MMR fail")
    mock_vector_store.similarity_search.side_effect = Exception("Sim fail")

    results = retrieve_relevant_chunks(mock_vector_store, query="No result", top_k=3)
    assert results == []


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_filter(mock_logger, mock_vector_store, mock_chunks):
    """test retrieve_relevant_chunks with paper_id filter."""
    mock_vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    mock_vector_store.max_marginal_relevance_search.return_value = mock_chunks

    results = retrieve_relevant_chunks(
        mock_vector_store, query="filter test", paper_ids=["P1"], top_k=3
    )
    assert results == mock_chunks
    args, kwargs = mock_vector_store.max_marginal_relevance_search.call_args
    assert len(args) == 0
    assert kwargs["filter"] == {"paper_id": ["P1"]}


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_success(
    mock_logger, mock_vector_store, mock_scored_chunks
):
    """retrieve_relevant_chunks_with_scores with GPU path."""
    mock_vector_store.has_gpu = True
    mock_logger.debug = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = mock_scored_chunks

    results = retrieve_relevant_chunks_with_scores(
        vector_store=mock_vector_store,
        query="score test",
        top_k=5,
        score_threshold=0.85,
    )

    assert all(score >= 0.85 for _, score in results)
    assert len(results) == 2  # Only scores 0.9 and 0.95 pass


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_fallback_to_default(
    mock_logger, mock_vector_store, mock_chunks
):
    """test retrieve_relevant_chunks_with_scores fallback to default search."""
    del mock_vector_store.similarity_search_with_score
    mock_vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    mock_vector_store.max_marginal_relevance_search.return_value = mock_chunks

    results = retrieve_relevant_chunks_with_scores(
        vector_store=mock_vector_store,
        query="fallback default",
        top_k=4,
        score_threshold=0.5,
    )

    assert isinstance(results, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in results)


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_error_returns_empty(
    mock_logger, mock_vector_store
):
    """ "retrieve_relevant_chunks_with_scores handles errors gracefully."""
    mock_vector_store.similarity_search_with_score.side_effect = Exception(
        "score search failed"
    )
    mock_logger.debug = MagicMock()

    results = retrieve_relevant_chunks_with_scores(
        vector_store=mock_vector_store,
        query="error path",
        top_k=3,
        score_threshold=0.1,
    )

    assert results == []


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_no_vector_store(mock_logger):
    """error handling when vector store is not initialized."""
    result = retrieve_relevant_chunks(vector_store=None, query="irrelevant")
    assert result == []
    mock_logger.error.assert_called_with("Vector store is not initialized")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_no_vector_store(mock_logger):
    """check error handling when vector store is not initialized."""
    result = retrieve_relevant_chunks_with_scores(vector_store=None, query="none")
    assert result == []
    mock_logger.error.assert_called_with("Vector store is not initialized")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_default_search_params(
    mock_logger, mock_vector_store, mock_chunks
):
    """retrieve_relevant_chunks uses default search params when not set."""
    mock_vector_store.has_gpu = False
    delattr(mock_vector_store, "search_params")  # Simulate missing search_params
    mock_vector_store.max_marginal_relevance_search.return_value = mock_chunks

    results = retrieve_relevant_chunks(
        mock_vector_store,
        query="default search param test",
        top_k=5,
    )

    assert results == mock_chunks
    mock_logger.debug.assert_any_call(
        "Using default search parameters (no hardware optimization)"
    )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_paper_filter(
    mock_logger, mock_vector_store, mock_scored_chunks
):
    """ensure retrieve_relevant_chunks_with_scores applies paper_id filter."""
    mock_vector_store.similarity_search_with_score.return_value = mock_scored_chunks
    mock_logger.debug = MagicMock()

    results = retrieve_relevant_chunks_with_scores(
        vector_store=mock_vector_store,
        query="filtered score",
        paper_ids=["P123"],
        top_k=5,
        score_threshold=0.0,
    )

    assert isinstance(results, list)
    assert mock_vector_store.similarity_search_with_score.call_args[1]["filter"] == {
        "paper_id": ["P123"]
    }
