"""tests for the PDF batch processor module."""

from unittest.mock import MagicMock, patch

import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor import (
    add_papers_batch,
)


@pytest.fixture
def base_args():
    """base_args fixture to provide common arguments for tests."""
    return {
        "vector_store": MagicMock(),
        "loaded_papers": set(),
        "paper_metadata": {},
        "documents": {},
        "config": {"param": "value"},
        "metadata_fields": ["Title", "Author"],
        "has_gpu": False,
    }


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf"
)
def test_no_papers_to_add(mock_loader, base_args):
    """test case where no papers are provided to add."""
    add_papers_batch(papers_to_add=[], **base_args)
    mock_loader.assert_not_called()


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf"
)
def test_all_papers_already_loaded(mock_loader, base_args):
    """test case where all papers are already loaded."""
    base_args["loaded_papers"].update(["p1", "p2"])
    add_papers_batch(
        papers_to_add=[("p1", "url1", {}), ("p2", "url2", {})], **base_args
    )
    mock_loader.assert_not_called()


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf"
)
def test_successful_batch_embedding(mock_loader, base_args):
    """test case where papers are successfully loaded and embedded."""
    mock_loader.return_value = [
        MagicMock(page_content="Page 1"),
        MagicMock(page_content="Page 2"),
    ]

    mock_collection = MagicMock()
    mock_collection.num_entities = 2
    mock_collection.query.return_value = [{"paper_id": "p1"}]
    base_args["vector_store"].col = mock_collection

    add_papers_batch(
        papers_to_add=[("p1", "url1", {"Title": "Paper One"})], **base_args
    )

    assert "p1" in base_args["paper_metadata"]
    assert "p1" in base_args["loaded_papers"]
    base_args["vector_store"].add_documents.assert_called_once()
    mock_collection.flush.assert_called()


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf"
)
def test_pdf_loading_failure(mock_loader, base_args):
    """pdf loading fails, should not call vector store."""

    def fail_once(*args, **kwargs):
        """Simulate a failure in loading PDF."""
        raise RuntimeError("Failed to load PDF")

    mock_loader.side_effect = fail_once

    add_papers_batch(
        papers_to_add=[("p1", "url1", {"Title": "Paper One"})], **base_args
    )

    base_args["vector_store"].add_documents.assert_not_called()


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf"
)
def test_empty_chunks_after_loading(mock_loader, base_args):
    """test case where no chunks are returned after loading PDF."""
    mock_loader.return_value = []  # Simulate no chunks

    add_papers_batch(papers_to_add=[("p1", "url1", {})], **base_args)

    base_args["vector_store"].add_documents.assert_not_called()


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf"
)
def test_vector_store_insert_failure(mock_loader, base_args):
    """test case where vector store insertion fails."""
    mock_loader.return_value = [MagicMock(page_content="page")]

    def raise_error(*args, **kwargs):
        raise Exception("Vector store failed")

    base_args["vector_store"].add_documents.side_effect = raise_error

    mock_collection = MagicMock()
    base_args["vector_store"].col = mock_collection

    with pytest.raises(Exception, match="Vector store failed"):
        add_papers_batch(papers_to_add=[("p1", "url1", {})], **base_args)


@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.utils.batch_processor.load_and_split_pdf"
)
def test_verify_insert_failure_logging(mock_loader, base_args):
    """verify_insert_success should log error but not raise."""
    mock_loader.return_value = [MagicMock(page_content="page")]

    # Mock vector store and .col to raise error inside verify_insert_success
    mock_col = MagicMock()
    mock_col.flush.side_effect = RuntimeError("flush failed")
    base_args["vector_store"].col = mock_col

    base_args["vector_store"].add_documents = MagicMock()

    # Should not raise error from `verify_insert_success`, just log it
    add_papers_batch(
        papers_to_add=[("p1", "url1", {"Title": "fail test"})], **base_args
    )
