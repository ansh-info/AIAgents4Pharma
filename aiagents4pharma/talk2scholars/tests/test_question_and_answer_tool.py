"""
Unit tests for question_and_answer tool functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from aiagents4pharma.talk2scholars.tools.pdf.question_and_answer import (
    question_and_answer,
)


@pytest.fixture
def mock_input():
    return {
        "question": "What is the main contribution of the paper?",
        "tool_call_id": "test_tool_call_id",
        "state": {
            "article_data": {"paper1": {"title": "Test Paper", "pdf_url": "url1"}},
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
        },
    }


@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.format_answer")
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.retrieve_and_rerank_chunks"
)
@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_all_papers")
@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_hydra_config")
@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.QAToolHelper")
def test_question_and_answer_success(
    mock_helper_cls,
    mock_load_config,
    mock_load_all_papers,
    mock_retrieve_rerank,
    mock_format_answer,
    mock_input,
):
    # Mock config and helper
    mock_helper = MagicMock()
    mock_helper.get_state_models_and_data.return_value = (
        mock_input["state"]["text_embedding_model"],
        mock_input["state"]["llm_model"],
        mock_input["state"]["article_data"],
    )
    mock_helper.init_vector_store.return_value = MagicMock()
    mock_helper.has_gpu = True
    mock_helper_cls.return_value = mock_helper
    mock_load_config.return_value = {"config_key": "value"}

    # Mock reranked chunks and answer
    mock_retrieve_rerank.return_value = [{"chunk": "relevant content"}]
    mock_format_answer.return_value = "Here is your answer."

    result = question_and_answer.invoke(mock_input)

    assert isinstance(result, Command)
    assert "messages" in result.update
    assert isinstance(result.update["messages"][0], ToolMessage)
    assert result.update["messages"][0].content == "Here is your answer."
    assert result.update["messages"][0].tool_call_id == mock_input["tool_call_id"]


@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.format_answer")
@patch(
    "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.retrieve_and_rerank_chunks"
)
@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_all_papers")
@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_hydra_config")
@patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.QAToolHelper")
def test_question_and_answer_no_reranked_chunks(
    mock_helper_cls,
    mock_load_config,
    mock_load_all_papers,
    mock_retrieve_rerank,
    mock_format_answer,
    mock_input,
):
    # Mock helper
    mock_helper = MagicMock()
    mock_helper.get_state_models_and_data.return_value = (
        mock_input["state"]["text_embedding_model"],
        mock_input["state"]["llm_model"],
        mock_input["state"]["article_data"],
    )
    mock_helper.init_vector_store.return_value = MagicMock()
    mock_helper.has_gpu = False
    mock_helper_cls.return_value = mock_helper
    mock_load_config.return_value = {"config_key": "value"}

    # Mock no reranked chunks
    mock_retrieve_rerank.return_value = []
    mock_format_answer.return_value = "No relevant information found."

    result = question_and_answer.invoke(mock_input)

    assert isinstance(result, Command)
    assert "messages" in result.update
    assert isinstance(result.update["messages"][0], ToolMessage)
    assert result.update["messages"][0].content == "No relevant information found."
