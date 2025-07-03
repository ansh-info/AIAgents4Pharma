"""collection_manager for managing Milvus collections for PDF chunks."""

from unittest.mock import MagicMock, patch

import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils import collection_manager


@pytest.fixture
def config_mock():
    """config_mock fixture to simulate configuration."""

    class Config:
        class Milvus:
            embedding_dim = 768

        milvus = Milvus()

    return Config()


@pytest.fixture
def index_params():
    """index_params fixture to provide index parameters for tests."""
    return {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}


def test_cached_collection_returned(config_mock, index_params):
    """ "check if cached collection is returned."""
    mock_collection = MagicMock()
    collection_name = "test_cached"
    collection_manager._collection_cache[collection_name] = mock_collection

    result = collection_manager.ensure_collection_exists(
        collection_name, config_mock, index_params, has_gpu=False
    )

    assert result == mock_collection
    del collection_manager._collection_cache[collection_name]  # clean up


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_create_new_collection(
    mock_utility, mock_Collection, config_mock, index_params
):
    """check if new collection is created when it does not exist."""
    mock_utility.list_collections.return_value = []

    mock_collection = MagicMock()
    mock_Collection.return_value = mock_collection
    mock_collection.indexes = [MagicMock(field_name="embedding")]
    mock_collection.num_entities = 5

    result = collection_manager.ensure_collection_exists(
        "new_collection", config_mock, index_params, has_gpu=True
    )

    assert mock_collection.create_index.called
    assert mock_collection.load.called
    assert result == mock_collection


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_load_existing_collection(
    mock_utility, mock_Collection, config_mock, index_params
):
    """test loading an existing collection."""
    mock_utility.list_collections.return_value = ["existing_collection"]

    mock_collection = MagicMock()
    mock_Collection.return_value = mock_collection
    mock_collection.indexes = []
    mock_collection.num_entities = 0

    result = collection_manager.ensure_collection_exists(
        "existing_collection", config_mock, index_params, has_gpu=False
    )

    mock_collection.load.assert_called_once()
    assert result == mock_collection


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_debug_collection_state_failure(
    mock_utility, mock_Collection, config_mock, index_params
):
    """debug_collection_state should log but not raise on failure."""
    mock_utility.list_collections.return_value = ["bad_collection"]

    mock_collection = MagicMock()
    mock_Collection.return_value = mock_collection
    mock_collection.indexes = []
    mock_collection.num_entities = 10

    # Force failure inside debug_collection_state
    mock_collection.schema = property(
        lambda self: (_ for _ in ()).throw(Exception("bad schema"))
    )

    # Proceed with normal call (it will log but not raise)
    result = collection_manager.ensure_collection_exists(
        "bad_collection", config_mock, index_params, has_gpu=True
    )
    assert result == mock_collection


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_ensure_collection_exception(
    mock_utility, mock_Collection, config_mock, index_params
):
    """ensure_collection_exists should raise on utility failure."""
    mock_utility.list_collections.side_effect = RuntimeError("milvus failure")

    with pytest.raises(RuntimeError, match="milvus failure"):
        collection_manager.ensure_collection_exists(
            "fail_collection", config_mock, index_params, has_gpu=False
        )


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_debug_collection_state_logs_exception(
    mock_utility, mock_Collection, config_mock, index_params
):
    """collection_manager should log exceptions in debug_collection_state."""
    mock_utility.list_collections.return_value = ["trouble_collection"]

    mock_collection = MagicMock()
    mock_collection.num_entities = 5
    mock_collection.schema = "Fake schema"

    # Simulate exception when accessing .indexes
    type(mock_collection).indexes = property(
        lambda self: (_ for _ in ()).throw(Exception("Index fetch failed"))
    )

    mock_Collection.return_value = mock_collection

    # This should trigger the logger.error inside debug_collection_state
    result = collection_manager.ensure_collection_exists(
        "trouble_collection", config_mock, index_params, has_gpu=True
    )
    assert result == mock_collection
