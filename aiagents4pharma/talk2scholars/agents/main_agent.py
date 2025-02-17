# /usr/bin/env python3

"""
Agent for interacting with Semantic Scholar
"""

import logging
import hydra
from typing import Annotated
from pydantic import Field
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import HumanMessage, ToolMessage
from ..state.state_talk2scholars import Talk2Scholars
from ..agents import s2_agent

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool(parse_docstring=True)
def s2_tool(tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict, InjectedState],
            task: str = Field(
                        description="The task to be performed by the tool"
                        "e.g. 'Find papers about AI agents',"
                        " 'Get recommendations for this paper'"
                        " 'Get recommendations for multiple papers'")
                        ) -> Command:
    """
    This tool can be used to search or get recommendations for papers.

    Args:
        tool_call_id (str): The unique identifier for the tool call.
        state (dict): The state of the main agent.
        task (str): The task to be performed by the tool.

    Returns:
        Command: The command to be executed by the agent.
    """
    logger.log(logging.INFO, "Searching for papers on %s", task)
    s2_agent_get_app = s2_agent.get_app("thread_123", "gpt-4o-mini")
    config = {"configurable": {"thread_id": "thread_123"}}
    # Updte the state of s2 agent with the state of main agent
    s2_agent_get_app.update_state(config,
                                  {"papers": state["papers"],
                                    "multi_papers": state["multi_papers"],
                                    "last_displayed_papers": state["last_displayed_papers"]})
    # Invoke the s2 agent
    s2_agent_get_app.invoke(
                        {"messages": [HumanMessage(content=task)]},
                        config = config
                    )
    # Get the current state of s2 agent
    current_state = s2_agent_get_app.get_state(config)
    # Prepare tool messages list
    # This will contain the artifacts of the tools of s2 agent
    # to be used by the main agent
    tool_artifacts = {}
    tool_contents = {}
    for msg in current_state.values["messages"]:
        if isinstance(msg, ToolMessage):
            if not msg.artifact:
                continue
            tool_artifacts[msg.name] = msg.artifact
            tool_contents[msg.name] = msg.content
    return Command(
            update={
                    "papers": current_state.values["papers"],
                    "multi_papers": current_state.values["multi_papers"],
                    "last_displayed_papers": current_state.values["last_displayed_papers"],
                    "messages":[
                                ToolMessage(
                                    content=tool_contents,
                                    tool_call_id=tool_call_id,
                                    artifact=tool_artifacts
                                )
                            ]
            },
        )

def get_app(uniq_id, llm_model="gpt-4o-mini"):
    """
    This function returns the langraph app for the Main agent.
    """
    def agent_main_node(state: Talk2Scholars) -> Command:
        """
        This function calls the model and always returns to supervisor.
        """
        logger.log(logging.INFO, "Creating Agent_Main node with thread_id %s", uniq_id)
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    logger.log(logging.INFO, "thread_id, llm_model: %s, %s", uniq_id, llm_model)
    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for the main agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.s2_agent

    # Define the tools
    tools = ToolNode([s2_tool])

    # Define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    # Create the agent
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Scholars,
        state_modifier=cfg.s2_agent,
        checkpointer=MemorySaver(),
    )

    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_main", agent_main_node)
    workflow.add_edge(START, "agent_main")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)
    logger.log(logging.INFO, "Compiled the graph")

    return app
