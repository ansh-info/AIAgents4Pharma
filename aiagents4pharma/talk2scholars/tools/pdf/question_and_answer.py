"""
LangGraph PDF Retrieval-Augmented Generation (RAG) Tool

This tool answers user questions using the traditional RAG pipeline:
1. Retrieve relevant chunks from ALL papers in the vector store
2. Rerank chunks using NVIDIA NIM reranker to find the most relevant ones
3. Generate answer using the top reranked chunks

Traditional RAG Pipeline Flow:
  Query → Retrieve chunks from ALL papers → Rerank chunks → Generate answer

This ensures the best possible chunks are selected across all available papers,
not just from pre-selected papers.
"""

import logging
import os
import time
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

import hydra
from .utils.get_vectorstore import get_vectorstore
from .utils.paper_loader import load_all_papers
from .utils.rag_pipeline import retrieve_and_rerank_chunks
from .utils.answer_formatter import format_answer

# Configuration and helper will be initialized in the tool function

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def load_hydra_config() -> Any:
    """
    Load the configuration using Hydra and return the configuration for the Q&A tool.
    """
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tools/question_and_answer=default"],
        )
        config = cfg.tools.question_and_answer
        logger.debug("Loaded Question and Answer tool configuration.")
        return config


def _get_state_models_and_data(
    state: dict, call_id: str
) -> tuple[Any, Any, dict[str, Any]]:
    """Retrieve embedding model, LLM, and article data from agent state."""
    text_emb = state.get("text_embedding_model")
    if not text_emb:
        msg = "No text embedding model found in state."
        logger.error("%s: %s", call_id, msg)
        raise ValueError(msg)
    llm = state.get("llm_model")
    if not llm:
        msg = "No LLM model found in state."
        logger.error("%s: %s", call_id, msg)
        raise ValueError(msg)
    articles = state.get("article_data", {})
    if not articles:
        msg = "No article_data found in state."
        logger.error("%s: %s", call_id, msg)
        raise ValueError(msg)
    return text_emb, llm, articles


def _init_vector_store(emb_model: Any, call_id: str) -> Any:
    """Get the singleton Milvus vector store instance with GPU/CPU optimization."""
    logger.info(
        "%s: Getting singleton vector store instance with hardware optimization",
        call_id,
    )
    vs = get_vectorstore(embedding_model=emb_model, config=load_hydra_config())

    # Get GPU availability from vector store
    has_gpu = getattr(vs, "has_gpu", False)
    hardware_type = "GPU-accelerated" if has_gpu else "CPU-only"

    logger.info(
        "%s: Vector store initialized (%s mode)",
        call_id,
        hardware_type,
    )

    # Log hardware-specific configuration
    if hasattr(vs, "index_params"):
        index_type = vs.index_params.get("index_type", "Unknown")
        logger.info(
            "%s: Using %s index type for %s processing",
            call_id,
            index_type,
            hardware_type,
        )

    return vs


class QuestionAndAnswerInput(BaseModel):
    """
    Pydantic schema for the PDF Q&A tool inputs.

    Fields:
      question: User's free-text query to answer based on PDF content.
      tool_call_id: LangGraph-injected call identifier for tracking.
      state: Shared agent state dict containing:
        - article_data: metadata mapping of paper IDs to info (e.g., 'pdf_url', title).
        - text_embedding_model: embedding model instance for chunk indexing.
        - llm_model: chat/LLM instance for answer generation.
    """

    question: str = Field(
        description="User question for generating a PDF-based answer."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool(args_schema=QuestionAndAnswerInput, parse_docstring=True)
def question_and_answer(
    question: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    LangGraph tool for Retrieval-Augmented Generation over PDFs using traditional RAG pipeline.

    Traditional RAG Pipeline Implementation:
      1. Load ALL available PDFs into Milvus vector store (if not already loaded)
      2. Retrieve relevant chunks from ALL papers using vector similarity search
      3. Rerank retrieved chunks using NVIDIA NIM semantic reranker
      4. Generate answer using top reranked chunks with source attribution

    This approach ensures the best chunks are selected across all available papers,
    rather than pre-selecting papers and potentially missing relevant information.

    Args:
      question (str): The free-text question to answer.
      state (dict): Injected agent state; must include:
        - article_data: mapping paper IDs → metadata (pdf_url, title, etc.)
        - text_embedding_model: embedding model instance.
        - llm_model: chat/LLM instance.
      tool_call_id (str): Internal identifier for this tool invocation.

    Returns:
      Command[Any]: updates conversation state with a ToolMessage(answer).

    Raises:
      ValueError: when required models or metadata are missing in state.
      RuntimeError: when no relevant chunks can be retrieved for the query.
    """
    call_id = f"qa_call_{time.time()}"
    logger.info(
        "Starting PDF Question and Answer tool (Traditional RAG Pipeline) - Call %s",
        call_id,
    )
    logger.info("%s: Question: '%s'", call_id, question)

    # Extract models and article metadata directly
    text_emb, llm_model, article_data = _get_state_models_and_data(state, call_id)

    # Initialize or reuse Milvus vector store directly
    logger.info("%s: Initializing vector store", call_id)
    vs = _init_vector_store(text_emb, call_id)

    # Load ALL papers (traditional RAG approach)
    logger.info(
        "%s: Loading all %d papers into vector store (traditional RAG approach)",
        call_id,
        len(article_data),
    )
    # Get GPU status from vector store
    has_gpu = getattr(vs, "has_gpu", False)

    load_all_papers(
        vector_store=vs,
        articles=article_data,
        call_id=call_id,
        config=load_hydra_config(),
        has_gpu=has_gpu,
    )

    # Traditional RAG Pipeline: Retrieve from ALL papers, then rerank
    logger.info(
        "%s: Starting traditional RAG pipeline: retrieve → rerank → generate",
        call_id,
    )

    # Retrieve and rerank chunks in one step
    reranked_chunks = retrieve_and_rerank_chunks(
        vs, question, load_hydra_config(), call_id, has_gpu
    )

    if not reranked_chunks:
        msg = f"No relevant chunks found for question: '{question}'"
        logger.warning("%s: %s", call_id, msg)

    # Generate answer using reranked chunks
    logger.info(
        "%s: Generating answer using %d reranked chunks",
        call_id,
        len(reranked_chunks),
    )
    response_text = format_answer(
        question,
        reranked_chunks,
        llm_model,
        article_data,
        load_hydra_config(),
        call_id=call_id,
        has_gpu=has_gpu,
    )

    logger.info(
        "%s: Successfully traditional completed RAG pipeline",
        call_id,
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=response_text,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
