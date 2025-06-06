"""
Updated Unit Tests for the Zotero agent (Zotero Library Managent sub-agent).
"""

# pylint: disable=redefined-outer-name
from unittest import mock
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from ..agents.zotero_agent import get_app
from ..state.state_talk2scholars import Talk2Scholars

LLM_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@pytest.fixture(autouse=True)
def mock_hydra_fixture():
    """Mock Hydra configuration to prevent external dependencies."""
    with mock.patch("hydra.initialize"), mock.patch("hydra.compose") as mock_compose:
        cfg_mock = mock.MagicMock()
        cfg_mock.agents.talk2scholars.zotero_agent.temperature = 0
        cfg_mock.agents.talk2scholars.zotero_agent.zotero_agent = "Test prompt"
        mock_compose.return_value = cfg_mock
        yield mock_compose


@pytest.fixture
def mock_tools_fixture():
    """Mock tools to prevent execution of real API calls."""
    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2.display_dataframe.display_dataframe"
        ) as mock_s2_display,
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2.query_dataframe.query_dataframe"
        ) as mock_s2_query_dataframe,
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.s2."
            "retrieve_semantic_scholar_paper_id."
            "retrieve_semantic_scholar_paper_id"
        ) as mock_s2_retrieve_id,
        mock.patch(
            "aiagents4pharma.talk2scholars.tools.zotero.zotero_read.zotero_read"
        ) as mock_zotero_query_dataframe,
    ):
        mock_s2_display.return_value = {"result": "Mock Display Result"}
        mock_s2_query_dataframe.return_value = {"result": "Mock Query Result"}
        mock_s2_retrieve_id.return_value = {"paper_id": "MockPaper123"}
        mock_zotero_query_dataframe.return_value = {"result": "Mock Search Result"}

        yield [
            mock_s2_display,
            mock_s2_query_dataframe,
            mock_s2_retrieve_id,
            mock_zotero_query_dataframe,
        ]


@pytest.mark.usefixtures("mock_hydra_fixture")
def test_zotero_agent_initialization():
    """Test that S2 agent initializes correctly with mock configuration."""
    thread_id = "test_thread"
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.zotero_agent.create_react_agent"
    ) as mock_create:
        mock_create.return_value = mock.Mock()
        app = get_app(thread_id, llm_model=LLM_MODEL)
        assert app is not None
        assert mock_create.called


def test_zotero_agent_invocation():
    """Test that the S2 agent processes user input and returns a valid response."""
    thread_id = "test_thread"
    mock_state = Talk2Scholars(messages=[HumanMessage(content="Find AI papers")])
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.zotero_agent.create_react_agent"
    ) as mock_create:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Here are some AI papers")],
            "papers": {"id123": "AI Research Paper"},
        }
        app = get_app(thread_id, llm_model=LLM_MODEL)
        result = app.invoke(
            mock_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint",
                }
            },
        )
        assert "messages" in result
        assert "papers" in result
        assert result["papers"]["id123"] == "AI Research Paper"


def test_zotero_agent_tools_assignment(request):
    """Ensure that the correct tools are assigned to the agent."""
    thread_id = "test_thread"
    mock_tools = request.getfixturevalue("mock_tools_fixture")
    with (
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.zotero_agent.create_react_agent"
        ) as mock_create,
        mock.patch(
            "aiagents4pharma.talk2scholars.agents.zotero_agent.ToolNode"
        ) as mock_toolnode,
    ):
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        mock_tool_instance = mock.Mock()
        mock_tool_instance.tools = mock_tools
        mock_toolnode.return_value = mock_tool_instance
        get_app(thread_id, llm_model=LLM_MODEL)
        assert mock_toolnode.called
        assert len(mock_tool_instance.tools) == 4


def test_s2_query_dataframe_tool():
    """Test if the query_dataframe tool is correctly utilized by the agent."""
    thread_id = "test_thread"
    mock_state = Talk2Scholars(
        messages=[HumanMessage(content="Query results for AI papers")]
    )
    with mock.patch(
        "aiagents4pharma.talk2scholars.agents.zotero_agent.create_react_agent"
    ) as mock_create:
        mock_agent = mock.Mock()
        mock_create.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [HumanMessage(content="Query results for AI papers")],
            "last_displayed_papers": {},
            "papers": {
                "query_dataframe": "Mock Query Result"
            },  # Ensure the expected key is inside 'papers'
            "multi_papers": {},
        }
        app = get_app(thread_id, llm_model=LLM_MODEL)
        result = app.invoke(
            mock_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint",
                }
            },
        )
        assert "query_dataframe" in result["papers"]
        assert mock_agent.invoke.called


def test_zotero_agent_hydra_failure():
    """Test exception handling when Hydra fails to load config."""
    thread_id = "test_thread"
    with mock.patch("hydra.initialize", side_effect=Exception("Hydra error")):
        with pytest.raises(Exception) as exc_info:
            get_app(thread_id, llm_model=LLM_MODEL)
        assert "Hydra error" in str(exc_info.value)
