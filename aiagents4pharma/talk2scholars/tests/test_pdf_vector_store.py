"""Vectorstore for managing PDF embeddings and similarity searches."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store import Vectorstore


@pytest.fixture
def mock_config():
    """mock configuration for testing."""

    class MockMilvusConfig:
        """mock configuration for Milvus connection."""

        host = "localhost"
        port = 19530
        collection_name = "test_collection"
        db_name = "test_db"
        embedding_dim = 384

    class MockGPUDetection:
        """mock configuration for GPU detection."""

        force_cpu_mode = False

    class MockConfig:
        """ "mock configuration class."""

        milvus = MockMilvusConfig()
        gpu_detection = MockGPUDetection()

    return MockConfig()


@pytest.fixture
def mock_embedding():
    """mock embedding model for testing."""
    return MagicMock(spec=Embeddings)


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
)
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists"
)
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu")
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config"
)
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
)
def test_vectorstore_initialization(
    mock_log_config,
    mock_get_index_config,
    mock_detect_gpu,
    mock_ensure_collection,
    mock_singleton_class,
    mock_config,
    mock_embedding,
):
    """test Vectorstore initialization with mock components."""
    mock_detect_gpu.return_value = True
    mock_get_index_config.return_value = ("mock_index_params", "mock_search_params")
    mock_singleton = MagicMock()
    mock_singleton.get_connection.return_value = None
    mock_singleton.get_vector_store.return_value = MagicMock()
    mock_singleton_class.return_value = mock_singleton

    mock_ensure_collection.return_value = "mock_collection"

    vs = Vectorstore(embedding_model=mock_embedding, config=mock_config)

    assert vs.embedding_model == mock_embedding
    assert vs.collection == "mock_collection"
    assert vs.has_gpu is True
    assert vs.vector_store == mock_singleton.get_vector_store.return_value


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
)
def test_similarity_search(mock_singleton_class, mock_embedding, mock_config):
    """test similarity search with mock vectorstore."""
    mock_vectorstore_instance = MagicMock()
    mock_vectorstore_instance.similarity_search.return_value = [
        Document(page_content="test content")
    ]
    mock_singleton = MagicMock()
    mock_singleton.get_vector_store.return_value = mock_vectorstore_instance
    mock_singleton.get_connection.return_value = None
    mock_singleton_class.return_value = mock_singleton

    with patch(
        "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists"
    ):
        with patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=False,
        ):
            with patch(
                "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
                return_value=("", ""),
            ):
                with patch(
                    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
                ):
                    vs = Vectorstore(mock_embedding, config=mock_config)
                    result = vs.similarity_search(query="test")
                    assert isinstance(result, list)
                    assert isinstance(result[0], Document)


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
)
def test_max_marginal_relevance_search(
    mock_singleton_class, mock_embedding, mock_config
):
    """test max marginal relevance search with mock vectorstore."""
    mock_vectorstore_instance = MagicMock()
    mock_vectorstore_instance.max_marginal_relevance_search.return_value = [
        Document(page_content="test content")
    ]
    mock_singleton = MagicMock()
    mock_singleton.get_vector_store.return_value = mock_vectorstore_instance
    mock_singleton.get_connection.return_value = None
    mock_singleton_class.return_value = mock_singleton

    with patch(
        "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists"
    ):
        with patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=False,
        ):
            with patch(
                "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
                return_value=("", ""),
            ):
                with patch(
                    "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
                ):
                    vs = Vectorstore(mock_embedding, config=mock_config)
                    result = vs.max_marginal_relevance_search(query="test")
                    assert isinstance(result, list)
                    assert isinstance(result[0], Document)


class DummyConfig:
    """dummy configuration class for testing purposes."""

    class Milvus:
        """mock configuration for Milvus connection."""

        host = "localhost"
        port = 19530
        db_name = "test_db"
        collection_name = "test_collection"
        embedding_dim = 768

    class GPUDetection:
        """force CPU mode for testing."""

        force_cpu_mode = True

    milvus = Milvus()
    gpu_detection = GPUDetection()


@pytest.fixture
def dummy_embedding():
    """dummy embedding model for testing purposes."""
    return MagicMock(spec=Embeddings)


@pytest.fixture
def dummy_config():
    """dummy configuration for testing purposes."""
    return DummyConfig()


@pytest.fixture
def dummy_vectorstore_components():
    """dummy vectorstore components for testing purposes."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=({"index_type": "IVF_FLAT"}, {"metric_type": "L2"}),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists",
            return_value=MagicMock(),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
        ) as mock_singleton_class,
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = "default"
        mock_singleton_class.return_value = mock_singleton
        yield mock_singleton, mock_vector_store


def test_force_cpu_mode(dummy_embedding, dummy_config, dummy_vectorstore_components):
    """force_cpu_mode should disable GPU detection."""
    dummy_config.gpu_detection.force_cpu_mode = True
    _, mock_vector_store = dummy_vectorstore_components
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    assert vs.has_gpu is False


def test_load_existing_papers_collection_missing(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """load_existing_papers should handle missing collection gracefully."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.col = None
    mock_vector_store.collection = None
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    vs._load_existing_paper_ids()
    assert isinstance(vs.loaded_papers, set)


def test_load_existing_papers_collection_empty(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """load_existing_papers should handle empty collection gracefully."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_collection = MagicMock()
    mock_collection.num_entities = 0
    mock_collection.flush.return_value = None
    mock_vector_store.col = mock_collection
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    vs._load_existing_paper_ids()
    assert len(vs.loaded_papers) == 0


def test_similarity_search_with_filters(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """test similarity search with filters."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.similarity_search.return_value = [Document(page_content="Test")]
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    results = vs.similarity_search(
        query="test query", k=2, filter={"paper_id": ["p1", "p2"], "page": 3}
    )
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_mmr_search_with_filters(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """mmr_search with filters should return filtered results."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.max_marginal_relevance_search.return_value = [
        Document(page_content="Test")
    ]
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    results = vs.max_marginal_relevance_search(
        query="test query", k=2, filter={"chunk_id": "c123"}
    )
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_filter_expression_all_cases(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """filter expression should handle all cases."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.similarity_search.return_value = [Document(page_content="Test")]

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store

    filter_dict = {
        "string_field": "text",
        "list_field": ["val1", "val2"],
        "int_field": 42,
        "float_field": 3.14,
    }

    results = vs.similarity_search(query="test", k=2, filter=filter_dict)
    assert len(results) == 1
    assert isinstance(results[0], Document)

    # Also test for MMR path
    mock_vector_store.max_marginal_relevance_search.return_value = [
        Document(page_content="Test")
    ]
    results_mmr = vs.max_marginal_relevance_search(
        query="test", k=2, filter=filter_dict
    )
    assert len(results_mmr) == 1
    assert isinstance(results_mmr[0], Document)


def test_load_existing_papers_with_entities(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Ensure existing papers are loaded when LangChain collection has entities."""
    _, mock_vector_store = dummy_vectorstore_components

    mock_collection = MagicMock()
    mock_collection.num_entities = 3
    mock_collection.flush.return_value = None
    mock_collection.query.return_value = [
        {"paper_id": "p1"},
        {"paper_id": "p2"},
        {"paper_id": "p3"},
    ]
    mock_vector_store.col = mock_collection

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store

    # Call private method explicitly to test loading logic
    vs._load_existing_paper_ids()

    assert len(vs.loaded_papers) == 3
    assert "p1" in vs.loaded_papers
    assert "p2" in vs.loaded_papers
    assert "p3" in vs.loaded_papers
