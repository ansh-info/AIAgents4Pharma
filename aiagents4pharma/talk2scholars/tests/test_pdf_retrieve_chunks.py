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
    """Fixture to simulate a vector store."""
    return MagicMock()


@pytest.fixture
def mock_chunks():
    """Fixture to simulate PDF chunks."""
    return [
        Document(page_content=f"chunk {i}", metadata={"paper_id": f"P{i%2}"})
        for i in range(5)
    ]


@pytest.fixture
def mock_scored_chunks():
    """Fixture to simulate scored PDF chunks."""
    return [
        (Document(page_content=f"chunk {i}", metadata={}), score)
        for i, score in enumerate([0.9, 0.8, 0.4, 0.95])
    ]


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_cpu_success(mock_logger, request):
    """Test retrieve_relevant_chunks with CPU path."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(vector_store, query="AI", top_k=5)

    assert results == chunks
    vector_store.max_marginal_relevance_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_gpu_success(mock_logger, request):
    """Test retrieve_relevant_chunks with GPU path."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = True
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(vector_store, query="AI", top_k=5)

    assert results == chunks
    vector_store.max_marginal_relevance_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_fallback_on_exception(mock_logger, request):
    """Test fallback to similarity_search on MMR failure."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.side_effect = Exception("MMR fail")
    vector_store.similarity_search.return_value = chunks

    results = retrieve_relevant_chunks(vector_store, query="Fallback test", top_k=3)
    assert results == chunks
    vector_store.similarity_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_final_failure_returns_empty(mock_logger, request):
    """Test that both MMR and similarity_search failures return empty list."""
    vector_store = request.getfixturevalue("mock_vector_store")
    vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.side_effect = Exception("MMR fail")
    vector_store.similarity_search.side_effect = Exception("Sim fail")

    results = retrieve_relevant_chunks(vector_store, query="No result", top_k=3)
    assert results == []


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_filter(mock_logger, request):
    """Test retrieve_relevant_chunks with paper_id filter."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(
        vector_store, query="filter test", paper_ids=["P1"], top_k=3
    )
    assert results == chunks
    args, kwargs = vector_store.max_marginal_relevance_search.call_args
    assert len(args) == 0
    assert kwargs["filter"] == {"paper_id": ["P1"]}


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_success(mock_logger, request):
    """Test retrieve_relevant_chunks_with_scores filters scores correctly."""
    vector_store = request.getfixturevalue("mock_vector_store")
    scored_chunks = request.getfixturevalue("mock_scored_chunks")
    vector_store.has_gpu = True
    mock_logger.debug = MagicMock()
    vector_store.similarity_search_with_score.return_value = scored_chunks

    results = retrieve_relevant_chunks_with_scores(
        vector_store=vector_store,
        query="score test",
        top_k=5,
        score_threshold=0.85,
    )

    assert all(score >= 0.85 for _, score in results)
    assert len(results) == 2


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_fallback_to_default(mock_logger, request):
    """Test retrieve_relevant_chunks_with_scores falls back if no score method."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    del vector_store.similarity_search_with_score
    vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks_with_scores(
        vector_store=vector_store,
        query="fallback default",
        top_k=4,
        score_threshold=0.5,
    )

    assert isinstance(results, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in results)


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_error_returns_empty(mock_logger, request):
    """Test score-based search failure returns empty list."""
    vector_store = request.getfixturevalue("mock_vector_store")
    vector_store.similarity_search_with_score.side_effect = Exception(
        "score search failed"
    )
    mock_logger.debug = MagicMock()

    results = retrieve_relevant_chunks_with_scores(
        vector_store=vector_store,
        query="error path",
        top_k=3,
        score_threshold=0.1,
    )

    assert results == []


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_no_vector_store(mock_logger):
    """Test when vector store is None."""
    result = retrieve_relevant_chunks(vector_store=None, query="irrelevant")
    assert result == []
    mock_logger.error.assert_called_with("Vector store is not initialized")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_no_vector_store(mock_logger):
    """Test retrieve_relevant_chunks_with_scores when vector store is None."""
    result = retrieve_relevant_chunks_with_scores(vector_store=None, query="none")
    assert result == []
    mock_logger.error.assert_called_with("Vector store is not initialized")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_default_search_params(mock_logger, request):
    """Test default search params used when not defined."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = False
    delattr(vector_store, "search_params")
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(
        vector_store,
        query="default search param test",
        top_k=5,
    )

    assert results == chunks
    mock_logger.debug.assert_any_call(
        "Using default search parameters (no hardware optimization)"
    )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_paper_filter(mock_logger, request):
    """Test retrieve_relevant_chunks_with_scores applies paper_id filter."""
    vector_store = request.getfixturevalue("mock_vector_store")
    scored_chunks = request.getfixturevalue("mock_scored_chunks")
    vector_store.similarity_search_with_score.return_value = scored_chunks
    mock_logger.debug = MagicMock()

    results = retrieve_relevant_chunks_with_scores(
        vector_store=vector_store,
        query="filtered score",
        paper_ids=["P123"],
        top_k=5,
        score_threshold=0.0,
    )

    assert isinstance(results, list)
    assert vector_store.similarity_search_with_score.call_args[1]["filter"] == {
        "paper_id": ["P123"]
    }
