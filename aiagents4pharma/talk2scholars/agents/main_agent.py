#!/usr/bin/env python3

"""
Main agent for the talk2scholars app using ReAct pattern.

This module implements a hierarchical agent system where a supervisor agent
routes queries to specialized sub-agents. It follows the LangGraph patterns
for multi-agent systems and implements proper state management.

The main components are:
1. Supervisor node with ReAct pattern for intelligent routing.
2. S2 agent node for handling academic paper queries.
3. Zotero agent node for processing paper queries from Zotero library.
4. Shared state management via Talk2Scholars.
5. Hydra-based configuration system.

Example:
    app = get_app("thread_123", "gpt-4o-mini")
    result = app.invoke({
        "messages": [("human", "Find papers about AI agents")]
    })
"""

import logging
from typing import Literal, Callable, TypedDict
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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


def make_supervisor_node(llm: BaseChatModel, thread_id: str) -> Callable:
    """Creates supervisor node for routing."""
    logger.info("Loading Hydra configuration for Talk2Scholars main agent.")
    cfg = get_hydra_config()
    logger.info("Hydra configuration loaded with values: %s", cfg)
    members = ["s2_agent", "zotero_agent"]
    options = ["FINISH"] + members
    # system_prompt = cfg.main_agent  # Use existing Hydra config
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

        # next: Literal["s2_agent", "zotero_agent", "FINISH"]

    def supervisor_node(
        state: Talk2Scholars,
    ) -> Command:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state[
            "messages"
        ]

        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        print ("GOTO: ", goto)
        if goto == "FINISH":
            print ("GOTO: ", goto)
            goto = END  # Using END from langgraph.graph

        return Command(goto=goto)

    return supervisor_node


def get_app(thread_id: str, llm_model: str = "gpt-4o-mini") -> StateGraph:
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
        logger.info("Calling S2 agent with state: %s", state)
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
        logger.info("S2 agent completed with response: %s", response)
        # import sys
        # sys.exit()

        # return Command(
        #     # goto=END,
        #     goto="supervisor",
        #     update={
        #         "messages": response["messages"],
        #         "papers": response.get("papers", {}),
        #         "multi_papers": response.get("multi_papers", {}),
        #     },
        # )
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="s2_agent"
                    )
                ]
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
        logger.info("Calling Zotero agent with state: %s", state)
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
        logger.info("Zotero agent completed with response: %s", response)

        # return Command(
        #     goto=END,
        #     update={
        #         "messages": response["messages"],
        #         "zotero_papers": response.get("zotero_papers", {}),
        #     },
        # )
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="zotero_agent"
                    )
                ]
            },
            # Always return to supervisor
            goto="supervisor",
        )

    # Initialize LLM
    logger.info("Using OpenAI model %s with temperature %s", llm_model, cfg.temperature)
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    # Build the graph
    workflow = StateGraph(Talk2Scholars)
    supervisor = make_supervisor_node(llm, thread_id)

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)
    workflow.add_node("zotero_agent", call_zotero_agent)

    # Only supervisor can decide to END
    workflow.add_edge(START, "supervisor")
    # workflow.add_edge("supervisor", "s2_agent")
    # workflow.add_edge("supervisor", "zotero_agent")
    # workflow.add_edge("supervisor", END)

    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
