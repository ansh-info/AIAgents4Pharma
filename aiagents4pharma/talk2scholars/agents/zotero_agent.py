#!/usr/bin/env python3

"""
Agent for interacting with Zotero
"""

import logging
import hydra
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.zotero.zotero_read import zotero_search_tool

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_app(uniq_id, llm_model="gpt-4o-mini"):
    """
    This function returns the langraph app.
    """

    def agent_zotero_node(state: Talk2Scholars):
        """
        This function calls the model.
        """
        logger.log(
            logging.INFO, "Creating Agent_Zotero node with thread_id %s", uniq_id
        )
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        return response

    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars Zotero agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["agents/talk2scholars/zotero_agent=default"],
        )
        cfg = cfg.agents.talk2scholars.zotero_agent

    # Define the tools
    tools = ToolNode([zotero_search_tool])

    # Define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    # Create the agent
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Scholars,
        state_modifier=cfg.zotero_agent,
        checkpointer=MemorySaver(),
    )

    # Define a new graph
    workflow = StateGraph(Talk2Scholars)

    # Define the node we will use
    workflow.add_node("agent_zotero", agent_zotero_node)

    # Set the entrypoint as `agent`
    workflow.add_edge(START, "agent_zotero")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)
    logger.log(logging.INFO, "Compiled the graph")

    return app
