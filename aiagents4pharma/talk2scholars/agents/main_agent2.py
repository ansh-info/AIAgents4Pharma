#!/usr/bin/env python3

"""
Main agent for the talk2scholars app using ReAct pattern.

This module implements a hierarchical agent system where a supervisor agent
routes queries to specialized sub-agents. It follows the LangGraph patterns
for multi-agent systems and implements proper state management.
"""

import logging
from typing import Literal, Callable
from pydantic import BaseModel, Field
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from ..agents import s2_agent
from ..agents import zotero_agent
from ..state.state_talk2scholars import Talk2Scholars

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hydra_config():
    """
    Loads and returns the Hydra configuration for the main agent.

    This function fetches the configuration settings for the Talk2Scholars
    agent, ensuring that all required parameters are properly initialized.

    Returns:
        Any: The configuration object for the main agent.
    """
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
        )
    return cfg.agents.talk2scholars.main_agent


def make_supervisor_node(llm_model: BaseChatModel,
                         thread_id: str) -> Callable:
    """Creates supervisor node for routing."""
    logger.info("Loading Hydra configuration for Talk2Scholars main agent.")
    cfg = get_hydra_config()
    logger.info("Hydra configuration loaded with values: %s", cfg)
    members = ["s2_agent", "zotero_agent"]
    options = ["FINISH"] + members
    # Define system prompt for general interactions
    system_prompt = cfg.system_prompt
    # Define router prompt for routing to sub-agents
    router_prompt = cfg.router_prompt

    class Router(BaseModel):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(
        state: Talk2Scholars,
    ) -> Command:
        messages = [SystemMessage(content=router_prompt)] + state["messages"]
        structured_llm = llm_model.with_structured_output(Router)
        response = structured_llm.invoke(messages)
        goto = response.next
        # if "next" in response:
        #     goto = response["next"]
        # else:
        #     goto = response["properties"]["next"]
        print("GOTO: ", goto)
        if goto == "FINISH":
            print("GOTO: ", goto)
            goto = END  # Using END from langgraph.graph
            # If no agents were called, and the last message was
            # from the user, call the LLM to respond to the user
            # with a slightly different system prompt.
            if isinstance(messages[-1], HumanMessage):
                response = llm_model.invoke(
                    [SystemMessage(content=system_prompt),]
                    + messages[1:]
                )
                return Command(
                    goto=goto, update={"messages": AIMessage(content=response.content)}
                )
        # Go to the requested agent
        return Command(goto=goto)

    return supervisor_node

def get_app(thread_id: str,
            llm_model: BaseChatModel = ChatOpenAI(model='gpt-4o-mini', temperature=0)) -> StateGraph:
    """
    Initializes and returns the LangGraph application with a hierarchical agent system.

    This function sets up the full agent architecture, including the supervisor
    and sub-agents, and compiles the LangGraph workflow for handling user queries.

    Args:
        thread_id (str): Unique identifier for the conversation session.
        llm_model (str, optional): The language model to be used. Defaults to "gpt-4o-mini".

    Returns:
        StateGraph: A compiled LangGraph application ready for query invocation.

    Example:
        app = get_app("thread_123")
        result = app.invoke(initial_state)
    """
    cfg = get_hydra_config()

    def call_s2_agent(
        state: Talk2Scholars,
    ) -> Command[Literal["supervisor"]]:
        """
        Calls the Semantic Scholar (S2) agent to process academic paper queries.

        This function invokes the S2 agent, retrieves relevant research papers,
        and updates the conversation state accordingly.

        Args:
            state (Talk2Scholars): The current conversation state, including user queries
                and any previously retrieved papers.

        Returns:
            Command: The next action to execute, along with updated messages and papers.

        Example:
            result = call_s2_agent(current_state)
            next_step = result.goto
        """
        logger.info("Calling S2 agent")
        app = s2_agent.get_app(thread_id, llm_model)

        # Invoke the S2 agent, passing state,
        # Pass both config_id and thread_id
        response = app.invoke(
            state,
            {
                "configurable": {
                    "config_id": thread_id,
                    "thread_id": thread_id,
                }
            },
        )
        logger.info("S2 agent completed with response")
        return Command(
            update={
                "messages": response["messages"],
                "papers": response.get("papers", {}),
                "multi_papers": response.get("multi_papers", {}),
                "last_displayed_papers": response.get("last_displayed_papers", {}),
            },
            # Always return to supervisor
            goto="supervisor",
        )

    def call_zotero_agent(
        state: Talk2Scholars,
    ) -> Command[Literal["supervisor"]]:
        """
        Calls the Zotero agent to process paper queries from Zotero library.

        This function invokes the Zotero agent, retrieves papers from Zotero,
        and updates the conversation state accordingly.

        Args:
            state (Talk2Scholars): The current conversation state, including user queries
                and any previously retrieved papers.

        Returns:
            Command: The next action to execute, along with updated messages and papers.

        Example:
            result = call_zotero_agent(current_state)
            next_step = result.goto
        """
        logger.info("Calling Zotero agent")
        app = zotero_agent.get_app(thread_id, llm_model)
        # Invoke the Zotero agent, passing state
        response = app.invoke(
            state,
            {
                "configurable": {
                    "config_id": thread_id,
                    "thread_id": thread_id,
                }
            },
        )
        logger.info("Zotero agent completed")
        return Command(
            update={
                "messages": response["messages"],
                "zotero_papers": response.get("zotero_papers", {}),
                "last_displayed_papers": response.get("last_displayed_papers", {}),
            },
            # Always return to supervisor
            goto="supervisor",
        )

    # Initialize LLM
    logger.info("Using model %s with temperature %s", llm_model, cfg.temperature)

    # Build the graph
    workflow = StateGraph(Talk2Scholars)
    supervisor = make_supervisor_node(llm_model, thread_id)
    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)
    workflow.add_node("zotero_agent", call_zotero_agent)
    # Add edges
    workflow.add_edge(START, "supervisor")
    # Compile the workflow
    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
