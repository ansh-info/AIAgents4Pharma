# /usr/bin/env python3

"""
Agent for interacting with Semantic Scholar
"""

import logging
import hydra

from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.s2.search import search_tool as s2_search
from ..tools.s2.display_results import display_results as s2_display
from ..tools.s2.query_results import query_results as s2_query_results
from ..tools.s2.single_paper_rec import (
    get_single_paper_recommendations as s2_single_rec,
)
from ..tools.s2.multi_paper_rec import (
    get_multi_paper_recommendations as s2_multi_rec
)
from langgraph.types import Command

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model="gpt-4o-mini"):
    """
    This function returns the langraph app for the S2 agent.
    """
    # def agent_s2_node(state: Talk2Scholars) -> Command[Literal["supervisor"]]:
    def agent_s2_node(state: Talk2Scholars) -> Command:
        """
        This function calls the model and always returns to supervisor.
        """
        logger.log(logging.INFO, "Creating Agent_S2 node with thread_id %s", uniq_id)
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    logger.log(logging.INFO, "thread_id, llm_model: %s, %s", uniq_id, llm_model)
    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars S2 agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/s2_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.s2_agent

    # Define the tools
    # tools = ToolNode([s2_search, s2_display, s2_single_rec, s2_multi_rec])
    tools = ToolNode([s2_search,
                      s2_display,
                      s2_query_results,
                      s2_single_rec,
                      s2_multi_rec])

    # Define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    # Create the agent
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Scholars,
        # prompt=cfg.s2_agent,
        state_modifier=cfg.s2_agent,
        checkpointer=MemorySaver(),
    )

    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_s2", agent_s2_node)
    workflow.add_edge(START, "agent_s2")
    # workflow.add_edge("agent_s2", "supervisor")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)
    logger.log(logging.INFO, "Compiled the graph")

    return app
