"""
Helper class for PDF Q&A tool orchestration: state validation, vectorstore init,
paper loading, chunk retrieval, reranking, and answer formatting.
Updated to follow traditional RAG pipeline: retrieve -> rerank -> generate
"""

import logging
from typing import Any, Dict, List, Optional

from .generate_answer import generate_answer
from .nvidia_nim_reranker import rerank_chunks
from .retrieve_chunks import retrieve_relevant_chunks
from .vector_store import Vectorstore
from .vector_store_manager import vector_store_manager

logger = logging.getLogger(__name__)


class QAToolHelper:
    """Encapsulates helper routines for the PDF Question & Answer tool."""

    def __init__(self) -> None:
        self.vector_store: Optional[Vectorstore] = None
        self.config: Any = None
        self.call_id: str = ""
        logger.debug("Initialized QAToolHelper")

    def start_call(self, config: Any, call_id: str) -> None:
        """Initialize helper with current config and call identifier."""
        self.config = config
        self.call_id = call_id
        logger.debug("QAToolHelper started call %s", call_id)

    def get_state_models_and_data(self, state: dict) -> tuple[Any, Any, Dict[str, Any]]:
        """Retrieve embedding model, LLM, and article data from agent state."""
        text_emb = state.get("text_embedding_model")
        if not text_emb:
            msg = "No text embedding model found in state."
            logger.error("%s: %s", self.call_id, msg)
            raise ValueError(msg)
        llm = state.get("llm_model")
        if not llm:
            msg = "No LLM model found in state."
            logger.error("%s: %s", self.call_id, msg)
            raise ValueError(msg)
        articles = state.get("article_data", {})
        if not articles:
            msg = "No article_data found in state."
            logger.error("%s: %s", self.call_id, msg)
            raise ValueError(msg)
        return text_emb, llm, articles

    def init_vector_store(self, emb_model: Any) -> Vectorstore:
        """Initialize or return existing Milvus vector store instance."""
        # First check if we have an existing instance
        self.vector_store = vector_store_manager.get_instance()

        if self.vector_store is None:
            # Initialize if not already done
            logger.info(
                "%s: No existing vector store found, initializing new instance",
                self.call_id,
            )
            self.vector_store = vector_store_manager.initialize(
                embedding_model=emb_model, config=self.config
            )
        else:
            logger.info("%s: Using existing shared vector store instance", self.call_id)

        # Get current stats
        stats = self.vector_store.get_collection_stats()
        logger.info(
            "%s: Vector store stats - Papers: %d, Entities: %s",
            self.call_id,
            stats.get("num_loaded_papers", 0),
            stats.get("num_entities", "unknown"),
        )

        return self.vector_store

    def load_all_papers(
        self,
        vs: Vectorstore,
        articles: Dict[str, Any],
    ) -> None:
        """Ensure all papers from article_data are loaded into the Milvus vector store."""
        papers_to_load = []

        # Check which papers need to be loaded
        for pid, article_info in articles.items():
            if pid not in vs.loaded_papers:
                pdf_url = article_info.get("pdf_url")
                if pdf_url:
                    papers_to_load.append(pid)
                else:
                    logger.warning(
                        "%s: No PDF URL found for paper %s", self.call_id, pid
                    )

        if not papers_to_load:
            logger.info(
                "%s: All %d papers already loaded in Milvus",
                self.call_id,
                len(articles),
            )
            return

        logger.info(
            "%s: Loading %d new papers into Milvus (already loaded: %d)",
            self.call_id,
            len(papers_to_load),
            len(articles) - len(papers_to_load),
        )

        # Load each paper
        for pid in papers_to_load:
            pdf_url = articles[pid]["pdf_url"]
            try:
                vs.add_paper(pid, pdf_url, articles[pid])
                logger.info("%s: Successfully loaded paper %s", self.call_id, pid)
            except (IOError, ValueError) as exc:
                logger.warning("%s: Error loading paper %s: %s", self.call_id, pid, exc)

    def retrieve_and_rerank_chunks(
        self,
        vs: Vectorstore,
        query: str,
    ) -> List[Any]:
        """
        Traditional RAG pipeline: retrieve chunks from all papers, then rerank.

        Args:
            vs: Vector store instance
            query: User query

        Returns:
            List of reranked chunks
        """
        logger.info(
            "%s: Starting traditional RAG pipeline - retrieve then rerank", self.call_id
        )

        # Step 1: Retrieve chunks from ALL papers (cast wide net)
        initial_chunks_count = self.config.get("initial_retrieval_k", 100)

        logger.info(
            "%s: Step 1 - Retrieving top %d chunks from ALL papers",
            self.call_id,
            initial_chunks_count,
        )

        retrieved_chunks = retrieve_relevant_chunks(
            vs,
            query=query,
            paper_ids=None,  # No filter - retrieve from all papers
            top_k=initial_chunks_count,
            mmr_diversity=self.config.get("mmr_diversity", 0.8),
        )

        if not retrieved_chunks:
            logger.warning("%s: No chunks retrieved from vector store", self.call_id)
            return []

        logger.info(
            "%s: Retrieved %d chunks from %d unique papers",
            self.call_id,
            len(retrieved_chunks),
            len(
                set(
                    chunk.metadata.get("paper_id", "unknown")
                    for chunk in retrieved_chunks
                )
            ),
        )

        # Step 2: Rerank the retrieved chunks
        logger.info(
            "%s: Step 2 - Reranking %d chunks to get top %d",
            self.call_id,
            len(retrieved_chunks),
            self.config.top_k_chunks,
        )

        reranked_chunks = rerank_chunks(
            chunks=retrieved_chunks,
            query=query,
            config=self.config,
            top_k=self.config.top_k_chunks,
        )

        logger.info(
            "%s: Reranking complete. Final %d chunks from %d unique papers",
            self.call_id,
            len(reranked_chunks),
            len(
                set(
                    chunk.metadata.get("paper_id", "unknown")
                    for chunk in reranked_chunks
                )
            ),
        )

        return reranked_chunks

    def format_answer(
        self,
        question: str,
        chunks: List[Any],
        llm: Any,
        articles: Dict[str, Any],
    ) -> str:
        """Generate the final answer text with source attributions."""
        result = generate_answer(question, chunks, llm, self.config)
        answer = result.get("output_text", "No answer generated.")

        # Get unique paper titles for source attribution
        titles: Dict[str, str] = {}
        for pid in result.get("papers_used", []):
            if pid in articles:
                titles[pid] = articles[pid].get("Title", "Unknown paper")

        # Format sources
        if titles:
            srcs = "\n\nSources:\n" + "\n".join(f"- {t}" for t in titles.values())
        else:
            srcs = ""

        logger.info(
            "%s: Generated answer using %d chunks from %d papers",
            self.call_id,
            len(chunks),
            len(titles),
        )

        return f"{answer}{srcs}"
