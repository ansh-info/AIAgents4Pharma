import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pymilvus.exceptions import MilvusException

from aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager import (
    VectorstoreSingleton,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.get_vectorstore import (
    get_vectorstore,
)


def test_singleton_instance_identity():
    a = VectorstoreSingleton()
    b = VectorstoreSingleton()
    assert a is b


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.detect_nvidia_gpu"
)
def test_detect_gpu_once(mock_detect):
    mock_detect.return_value = True
    singleton = VectorstoreSingleton()
    singleton._gpu_detected = None  # Reset for test
    result = singleton.detect_gpu_once()
    assert result is True
    # Should not call detect_nvidia_gpu again
    result2 = singleton.detect_gpu_once()
    mock_detect.assert_called_once()


def test_get_event_loop_reuses_existing():
    singleton = VectorstoreSingleton()
    loop1 = singleton.get_event_loop()
    loop2 = singleton.get_event_loop()
    assert loop1 is loop2


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.connections")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.db")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.utility")
def test_get_connection_creates_connection(mock_util, mock_db, mock_conns):
    singleton = VectorstoreSingleton()
    mock_conns.has_connection.return_value = True
    mock_db.list_database.return_value = []
    conn_key = singleton.get_connection("localhost", 19530, "test_db")
    assert conn_key == "default"
    mock_conns.remove_connection.assert_called_once()
    mock_conns.connect.assert_called_once()
    mock_db.create_database.assert_called_once_with("test_db")
    mock_db.using_database.assert_called_once_with("test_db")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.Milvus")
def test_get_vector_store_creates_if_missing(mock_milvus):
    singleton = VectorstoreSingleton()
    singleton._vector_stores.clear()
    singleton._event_loops.clear()
    mock_embed = MagicMock()
    vs = singleton.get_vector_store("collection1", mock_embed, {"host": "localhost"})
    assert "collection1" in singleton._vector_stores
    mock_milvus.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.get_vectorstore.Vectorstore")
def test_get_vectorstore_factory(mock_vectorstore):
    mock_config = MagicMock()
    mock_config.milvus.collection_name = "demo"
    mock_config.milvus.embedding_dim = 768
    mock_embed = MagicMock()

    # Force new instance
    result1 = get_vectorstore(mock_embed, mock_config, force_new=True)
    assert result1 == mock_vectorstore.return_value

    # Reuse existing instance
    result2 = get_vectorstore(mock_embed, mock_config)
    assert result2 == result1


def test_get_vectorstore_force_new():
    with patch(
        "aiagents4pharma.talk2scholars.tools.pdf.utils.get_vectorstore.Vectorstore"
    ) as MockVectorstore:
        mock_vs1 = MagicMock(name="Vectorstore1")
        mock_vs2 = MagicMock(name="Vectorstore2")
        MockVectorstore.side_effect = [mock_vs1, mock_vs2]

        dummy_config = MagicMock()
        dummy_config.milvus.collection_name = "my_test_collection"
        dummy_config.milvus.embedding_dim = 768

        vs1 = get_vectorstore(mock_vs1, dummy_config)
        vs2 = get_vectorstore(mock_vs2, dummy_config, force_new=True)

        assert vs1 is mock_vs1
        assert vs2 is mock_vs2
        assert vs1 != vs2


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.connections.connect"
)
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.connections.has_connection"
)
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.db")
def test_get_connection_milvus_error(mock_db, mock_has_connection, mock_connect):
    # Ensure the singleton has no previous cached connection
    manager = VectorstoreSingleton()
    manager._connections.clear()

    # Mock behavior
    mock_has_connection.return_value = False
    mock_connect.side_effect = MilvusException("Connection failed")

    with pytest.raises(MilvusException, match="Connection failed"):
        manager.get_connection("localhost", 19530, "test_db")


def test_get_event_loop_creates_new_loop_on_closed():
    manager = VectorstoreSingleton()
    manager._event_loops.clear()

    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = True  # Simulate a closed loop

    with (
        patch("asyncio.get_event_loop", return_value=mock_loop),
        patch("asyncio.new_event_loop") as mock_new_loop,
        patch("asyncio.set_event_loop") as mock_set_loop,
    ):

        new_loop = MagicMock()
        mock_new_loop.return_value = new_loop

        result_loop = manager.get_event_loop()

        # Ensure the fallback logic was executed
        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(new_loop)
        assert result_loop == new_loop
