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
from .vector_store import get_vectorstore

logger = logging.getLogger(__name__)


class QAToolHelper:
    """Encapsulates helper routines for the PDF Question & Answer tool."""

    def __init__(self) -> None:
        self.vector_store: Optional[Any] = None
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

    def init_vector_store(self, emb_model: Any) -> Any:
        """Get the singleton Milvus vector store instance."""
        # Use factory to get singleton instance
        logger.info(
            "%s: Getting singleton vector store instance",
            self.call_id,
        )
        self.vector_store = get_vectorstore(
            embedding_model=emb_model, config=self.config
        )

        return self.vector_store

    def load_all_papers(
        self,
        vs: Any,
        articles: Dict[str, Any],
    ) -> None:
        """Ensure all papers from article_data are loaded into the Milvus vector store."""
        papers_to_load = []
        skipped_papers = []
        already_loaded = []

        # Check which papers need to be loaded
        for pid, article_info in articles.items():
            if pid not in vs.loaded_papers:
                pdf_url = article_info.get("pdf_url")
                if pdf_url:
                    # Prepare tuple for batch loading
                    papers_to_load.append((pid, pdf_url, article_info))
                else:
                    skipped_papers.append(pid)
            else:
                already_loaded.append(pid)

        # Log summary of papers status
        logger.info(
            "%s: Paper loading summary - Total: %d, Already loaded: %d, To load: %d, No PDF: %d",
            self.call_id,
            len(articles),
            len(already_loaded),
            len(papers_to_load),
            len(skipped_papers),
        )

        if skipped_papers:
            logger.warning(
                "%s: Skipping %d papers without PDF URLs: %s%s",
                self.call_id,
                len(skipped_papers),
                skipped_papers[:5],  # Show first 5
                "..." if len(skipped_papers) > 5 else "",
            )

        if not papers_to_load:
            logger.info(
                "%s: All papers with PDFs are already loaded in Milvus", self.call_id
            )
            return

        # Use batch loading with parallel processing for ALL papers at once
        try:
            # Configure parallel workers - use more workers for better parallelism
            max_workers = min(10, max(3, len(papers_to_load)))  # Increased workers
            batch_size = self.config.get("embedding_batch_size", 100)

            logger.info(
                "%s: Loading %d papers in ONE BATCH using %d parallel workers (batch size: %d)",
                self.call_id,
                len(papers_to_load),
                max_workers,
                batch_size,
            )

            # This should process ALL papers at once
            vs.add_papers_batch(
                papers_to_add=papers_to_load,
                max_workers=max_workers,
                batch_size=batch_size,
            )

            logger.info(
                "%s: Successfully completed batch loading of all %d papers",
                self.call_id,
                len(papers_to_load),
            )

        except Exception as exc:
            logger.error(
                "%s: Error during batch paper loading: %s",
                self.call_id,
                exc,
                exc_info=True,
            )

    def retrieve_and_rerank_chunks(
        self,
        vs: Any,
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
