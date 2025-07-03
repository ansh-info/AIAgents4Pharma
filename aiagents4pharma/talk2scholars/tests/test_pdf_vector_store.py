"""Unit tests for the Vectorstore class with GPU support and embedding normalization."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store import Vectorstore


@pytest.fixture(name="mock_config")
def _mock_config():
    return SimpleNamespace(
        milvus=SimpleNamespace(
            host="localhost",
            port=19530,
            collection_name="test_collection",
            db_name="test_db",
            embedding_dim=384,
        ),
        gpu_detection=SimpleNamespace(force_cpu_mode=False),
    )


@pytest.fixture(name="mock_embedding")
def _mock_embedding():
    return MagicMock(spec=Embeddings)


@pytest.fixture(name="dummy_embedding")
def _dummy_embedding():
    return MagicMock(spec=Embeddings)


@pytest.fixture(name="dummy_config")
def _dummy_config():
    return SimpleNamespace(
        milvus=SimpleNamespace(
            host="localhost",
            port=19530,
            db_name="test_db",
            collection_name="test_collection",
            embedding_dim=768,
        ),
        gpu_detection=SimpleNamespace(force_cpu_mode=False),
    )


@pytest.fixture(name="dummy_vectorstore_components")
def _dummy_vectorstore_components():
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=(
                {"index_type": "IVF_FLAT", "metric_type": "IP"},
                {"nprobe": 10},
            ),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists",
            return_value=MagicMock(),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
        ) as singleton_cls,
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = "connected"
        singleton_cls.return_value = mock_singleton
        yield mock_singleton, mock_vector_store


def test_vectorstore_initialization(mock_config, mock_embedding):
    """Test Vectorstore initialization with GPU and mocked dependencies."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration",
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.wrap_"
            "embedding_model_if_needed",
            return_value=mock_embedding,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists",
            return_value="mock_collection",
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton",
        ) as mock_singleton_cls,
    ):
        # set up singleton
        mock_singleton = MagicMock()
        mock_singleton.get_connection.return_value = None
        mock_singleton.get_vector_store.return_value = MagicMock()
        mock_singleton_cls.return_value = mock_singleton

        vs = Vectorstore(embedding_model=mock_embedding, config=mock_config)

        assert vs.embedding_model is mock_embedding
        assert vs.collection == "mock_collection"
        assert vs.has_gpu is True
        assert vs.vector_store is mock_singleton.get_vector_store.return_value


def test_get_embedding_info(mock_config, mock_embedding):
    """Test retrieval of embedding configuration info."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=({"metric_type": "IP", "index_type": "IVF"}, {}),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store."
            "wrap_embedding_model_if_needed",
            return_value=mock_embedding,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists",
            return_value="mock_collection",
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
        ) as mock_singleton_cls,
    ):
        mock_singleton = MagicMock()
        mock_singleton.get_connection.return_value = None
        mock_singleton.get_vector_store.return_value = MagicMock()
        mock_singleton_cls.return_value = mock_singleton

        vs = Vectorstore(embedding_model=mock_embedding, config=mock_config)
        info = vs.get_embedding_info()

        assert info["has_gpu"] is True
        assert info["use_cosine"] is True
        assert "original_model_type" in info
        assert "wrapped_model_type" in info
        assert "normalization_enabled" in info


def test_load_existing_papers_with_exception(mock_embedding, mock_config):
    """Test that load_existing_paper_ids handles exceptions gracefully."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store."
            "wrap_embedding_model_if_needed",
            return_value=mock_embedding,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
        ) as singleton_cls,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists"
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
        ),
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store.col.flush.side_effect = Exception("flush failed")
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        vs = Vectorstore(mock_embedding, config=mock_config)
        vs.vector_store = mock_vector_store
        # call via getattr to avoid protected‐access lint
        getattr(vs, "_load_existing_paper_ids")()

        assert isinstance(vs.loaded_papers, set)


def test_ensure_collection_loaded_with_entities(mock_embedding, mock_config):
    """Test ensure_collection_loaded loads data into memory when entities are present."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store."
            "wrap_embedding_model_if_needed",
            return_value=mock_embedding,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
        ) as singleton_cls,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists"
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
        ),
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.num_entities = 5
        mock_vector_store.col = mock_collection
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        vs = Vectorstore(mock_embedding, config=mock_config)
        vs.vector_store = mock_vector_store
        getattr(vs, "_ensure_collection_loaded")()

        assert mock_collection.load.called


def test_ensure_collection_loaded_handles_exception(mock_embedding, mock_config):
    """Test ensure_collection_loaded handles exceptions."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store."
            "wrap_embedding_model_if_needed",
            return_value=mock_embedding,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
        ) as singleton_cls,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists"
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
        ),
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.flush.side_effect = Exception("flush error")
        mock_vector_store.col = mock_collection
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        vs = Vectorstore(mock_embedding, config=mock_config)
        vs.vector_store = mock_vector_store
        getattr(vs, "_ensure_collection_loaded")()

        # no exception should propagate
        assert True


def test_force_cpu_mode_logs_override(mock_config, mock_embedding):
    """Test logging of forced CPU mode."""
    # nothing complicated here—just flip the flag
    mock_config.gpu_detection.force_cpu_mode = True
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store."
            "wrap_embedding_model_if_needed",
            return_value=mock_embedding,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.VectorstoreSingleton"
        ) as singleton_cls,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.ensure_collection_exists",
            return_value="mock_collection",
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.detect_nvidia_gpu",
            return_value=True,
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.log_index_configuration"
        ),
    ):
        mock_singleton = MagicMock()
        mock_singleton.get_connection.return_value = None
        mock_singleton.get_vector_store.return_value = MagicMock()
        singleton_cls.return_value = mock_singleton

        vs = Vectorstore(embedding_model=mock_embedding, config=mock_config)
        assert vs.has_gpu is False


def test_similarity_metric_override(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Test setting of use_cosine from config.similarity_metric."""
    # replace inline class with SimpleNamespace
    dummy_config.similarity_metric = SimpleNamespace(use_cosine=False)
    _, _mock_vector_store = dummy_vectorstore_components
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    assert vs.use_cosine is False


def test_load_existing_paper_ids_fallback_to_collection(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Test fallback to collection and warning if both attributes are missing."""
    _, mock_vector_store = dummy_vectorstore_components
    # remove both attrs
    for attr in ("col", "collection"):
        if hasattr(mock_vector_store, attr):
            delattr(mock_vector_store, attr)

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    getattr(vs, "_load_existing_paper_ids")()
    assert isinstance(vs.loaded_papers, set)


def test_load_existing_papers_collection_empty_logs(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Test logging when collection is empty."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_collection = MagicMock()
    mock_collection.num_entities = 0
    mock_collection.flush.return_value = None
    mock_vector_store.col = mock_collection

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    getattr(vs, "_load_existing_paper_ids")()
    assert len(vs.loaded_papers) == 0


def test_similarity_search_filter_paths(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Test full filter expression generation path in similarity search."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.similarity_search.return_value = [Document(page_content="test")]
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store

    filter_dict = {
        "field1": "value",
        "field2": [1, 2],
        "field3": 99,
        "field4": 3.14,
    }
    result = vs.similarity_search(query="text", filter=filter_dict)
    assert isinstance(result, list)


def test_mmr_search_filter_paths(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Test full filter expression generation path in MMR search."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.max_marginal_relevance_search.return_value = [
        Document(page_content="test")
    ]
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store

    filter_dict = {"f": "text", "g": ["a", "b"], "h": 7, "j": 3.3}
    result = vs.max_marginal_relevance_search(query="q", filter=filter_dict)
    assert isinstance(result, list)


def test_ensure_collection_loaded_no_col_and_no_collection(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Test fallback logic if both `col` and `collection` are missing."""
    _, mock_vector_store = dummy_vectorstore_components
    for attr in ("col", "collection"):
        if hasattr(mock_vector_store, attr):
            delattr(mock_vector_store, attr)

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    getattr(vs, "_ensure_collection_loaded")()
    assert True  # no crash


def test_ensure_collection_loaded_empty_logs(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """Test logging path when collection is empty in _ensure_collection_loaded."""
    _, mock_vector_store = dummy_vectorstore_components
    mock_collection = MagicMock()
    mock_collection.num_entities = 0
    mock_vector_store.col = mock_collection

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store
    getattr(vs, "_ensure_collection_loaded")()
    assert True
