"""
Unit tests for question_and_answer tool functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from aiagents4pharma.talk2scholars.tools.pdf.question_and_answer import (
    question_and_answer,
)


@pytest.fixture
def mock_input():
    """mock_input fixture to simulate input for the question_and_answer tool."""
    return {
        "question": "What is the main contribution of the paper?",
        "tool_call_id": "test_tool_call_id",
        "state": {
            "article_data": {"paper1": {"title": "Test Paper", "pdf_url": "url1"}},
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
        },
    }


@pytest.fixture
def mock_dependencies():
    """Fixture that patches all necessary dependencies for question_and_answer."""
    with (
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.format_answer"
        ) as mock_format,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.retrieve_and_rerank_chunks"
        ) as mock_rerank,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_all_papers"
        ) as mock_load_papers,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_hydra_config"
        ) as mock_config,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.QAToolHelper"
        ) as mock_helper_cls,
        patch(
            "aiagents4pharma.talk2scholars.tools.pdf.utils.tool_helper.get_vectorstore"
        ) as mock_get_vs,
    ):

        yield {
            "mock_get_vectorstore": mock_get_vs,
            "mock_helper_cls": mock_helper_cls,
            "mock_load_config": mock_config,
            "mock_load_all_papers": mock_load_papers,
            "mock_retrieve_rerank": mock_rerank,
            "mock_format_answer": mock_format,
        }


def test_question_and_answer_success(mock_dependencies, mock_input):
    """question_and_answer should return a ToolMessage with the answer."""
    mocks = mock_dependencies
    mock_helper = MagicMock()
    mock_helper.get_state_models_and_data.return_value = (
        mock_input["state"]["text_embedding_model"],
        mock_input["state"]["llm_model"],
        mock_input["state"]["article_data"],
    )
    mock_helper.init_vector_store.return_value = MagicMock()
    mock_helper.has_gpu = True

    mocks["mock_helper_cls"].return_value = mock_helper
    mocks["mock_load_config"].return_value = {"config_key": "value"}
    mocks["mock_get_vectorstore"].return_value = MagicMock()
    mocks["mock_retrieve_rerank"].return_value = [{"chunk": "relevant content"}]
    mocks["mock_format_answer"].return_value = "Here is your answer."

    result = question_and_answer.invoke(mock_input)

    assert isinstance(result.update["messages"][0], ToolMessage)
    assert result.update["messages"][0].content == "Here is your answer."


def test_question_and_answer_no_reranked_chunks(mock_dependencies, mock_input):
    """question_and_answer should return a ToolMessage with no relevant information found."""
    mocks = mock_dependencies
    mock_helper = MagicMock()
    mock_helper.get_state_models_and_data.return_value = (
        mock_input["state"]["text_embedding_model"],
        mock_input["state"]["llm_model"],
        mock_input["state"]["article_data"],
    )
    mock_helper.init_vector_store.return_value = MagicMock()
    mock_helper.has_gpu = False

    mocks["mock_helper_cls"].return_value = mock_helper
    mocks["mock_load_config"].return_value = {"config_key": "value"}
    mocks["mock_get_vectorstore"].return_value = MagicMock()
    mocks["mock_retrieve_rerank"].return_value = []
    mocks["mock_format_answer"].return_value = "No relevant information found."

    result = question_and_answer.invoke(mock_input)

    assert isinstance(result.update["messages"][0], ToolMessage)
    assert result.update["messages"][0].content == "No relevant information found."
