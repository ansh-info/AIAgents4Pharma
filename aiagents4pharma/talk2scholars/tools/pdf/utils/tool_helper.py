"""
Helper class for PDF Q&A tool orchestration: state validation, vectorstore init,
paper loading, chunk retrieval, reranking, and answer formatting.
Updated to follow traditional RAG pipeline: retrieve -> rerank -> generate
Enhanced with automatic GPU/CPU detection and optimization.
"""

import logging
from typing import Any, Dict, List

from .generate_answer import generate_answer
from .nvidia_nim_reranker import rerank_chunks
from .retrieve_chunks import retrieve_relevant_chunks
from .singleton_manager import get_vectorstore

logger = logging.getLogger(__name__)


class QAToolHelper:
    """
    Encapsulates helper routines for the PDF Question & Answer tool.
    Enhanced with automatic GPU/CPU detection and optimization.
    """

    def __init__(self) -> None:
        self.config: Any = None
        self.call_id: str = ""
        self.has_gpu: bool = False  # Track GPU availability
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
        """Get the singleton Milvus vector store instance with GPU/CPU optimization."""
        logger.info(
            "%s: Getting singleton vector store instance with hardware optimization",
            self.call_id,
        )
        vs = get_vectorstore(embedding_model=emb_model, config=self.config)

        # Track GPU availability from vector store
        self.has_gpu = getattr(vs, "has_gpu", False)
        hardware_type = "GPU-accelerated" if self.has_gpu else "CPU-only"

        logger.info(
            "%s: Vector store initialized (%s mode)",
            self.call_id,
            hardware_type,
        )

        # Log hardware-specific configuration
        if hasattr(vs, "index_params"):
            index_type = vs.index_params.get("index_type", "Unknown")
            logger.info(
                "%s: Using %s index type for %s processing",
                self.call_id,
                index_type,
                hardware_type,
            )

        return vs

    def load_all_papers(
        self,
        vs: Any,
        articles: Dict[str, Any],
    ) -> None:
        """
        Ensure all papers from article_data are loaded into the Milvus vector store.
        Optimized for GPU/CPU processing.
        """
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

        # Log summary of papers status with hardware info
        hardware_info = (
            f" (GPU acceleration: {'enabled' if self.has_gpu else 'disabled'})"
        )
        logger.info(
            "%s: Paper loading summary%s - Total: %d, Already loaded: %d, To load: %d, No PDF: %d",
            self.call_id,
            hardware_info,
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
        # Adjust parameters based on hardware capabilities
        try:
            if self.has_gpu:
                # GPU can handle more parallel processing
                max_workers = min(
                    12, max(4, len(papers_to_load))
                )  # More workers for GPU
                batch_size = self.config.get(
                    "embedding_batch_size", 2000
                )  # Larger batches for GPU
                logger.info(
                    "%s: Using GPU-optimized loading parameters: %d workers, batch size %d",
                    self.call_id,
                    max_workers,
                    batch_size,
                )
            else:
                # CPU - more conservative parameters
                max_workers = min(
                    8, max(3, len(papers_to_load))
                )  # Conservative for CPU
                batch_size = self.config.get(
                    "embedding_batch_size", 1000
                )  # Smaller batches for CPU
                logger.info(
                    "%s: Using CPU-optimized loading parameters: %d workers, batch size %d",
                    self.call_id,
                    max_workers,
                    batch_size,
                )

            logger.info(
                "%s: Loading %d papers in ONE BATCH using %d parallel workers (batch size: %d, %s)",
                self.call_id,
                len(papers_to_load),
                max_workers,
                batch_size,
                "GPU accelerated" if self.has_gpu else "CPU processing",
            )

            # This should process ALL papers at once with hardware optimization
            vs.add_papers_batch(
                papers_to_add=papers_to_load,
                max_workers=max_workers,
                batch_size=batch_size,
            )

            logger.info(
                "%s: Successfully completed batch loading of all %d papers with %s",
                self.call_id,
                len(papers_to_load),
                "GPU acceleration" if self.has_gpu else "CPU processing",
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
        Optimized for GPU/CPU hardware.

        Args:
            vs: Vector store instance
            query: User query

        Returns:
            List of reranked chunks
        """
        hardware_mode = "GPU-accelerated" if self.has_gpu else "CPU-optimized"
        logger.info(
            "%s: Starting traditional RAG pipeline - retrieve then rerank (%s)",
            self.call_id,
            hardware_mode,
        )

        # Step 1: Retrieve chunks from ALL papers (cast wide net)
        # Adjust initial retrieval count based on hardware
        if self.has_gpu:
            # GPU can handle larger initial retrieval efficiently
            initial_chunks_count = self.config.get(
                "initial_retrieval_k", 150
            )  # Increased for GPU
            mmr_diversity = self.config.get(
                "mmr_diversity", 0.75
            )  # Slightly more diverse for larger sets
        else:
            # CPU - use conservative settings
            initial_chunks_count = self.config.get(
                "initial_retrieval_k", 100
            )  # Original
            mmr_diversity = self.config.get("mmr_diversity", 0.8)  # Original

        logger.info(
            "%s: Step 1 - Retrieving top %d chunks from ALL papers (%s mode)",
            self.call_id,
            initial_chunks_count,
            hardware_mode,
        )

        retrieved_chunks = retrieve_relevant_chunks(
            vs,
            query=query,
            paper_ids=None,  # No filter - retrieve from all papers
            top_k=initial_chunks_count,
            mmr_diversity=mmr_diversity,
        )

        if not retrieved_chunks:
            logger.warning("%s: No chunks retrieved from vector store", self.call_id)
            return []

        logger.info(
            "%s: Retrieved %d chunks from %d unique papers using %s",
            self.call_id,
            len(retrieved_chunks),
            len(
                set(
                    chunk.metadata.get("paper_id", "unknown")
                    for chunk in retrieved_chunks
                )
            ),
            hardware_mode,
        )

        # Step 2: Rerank the retrieved chunks
        final_chunk_count = self.config.top_k_chunks
        logger.info(
            "%s: Step 2 - Reranking %d chunks to get top %d",
            self.call_id,
            len(retrieved_chunks),
            final_chunk_count,
        )

        reranked_chunks = rerank_chunks(
            chunks=retrieved_chunks,
            query=query,
            config=self.config,
            top_k=final_chunk_count,
        )

        # Log final results with hardware info
        final_papers = len(
            set(chunk.metadata.get("paper_id", "unknown") for chunk in reranked_chunks)
        )

        logger.info(
            "%s: Reranking complete using %s. Final %d chunks from %d unique papers",
            self.call_id,
            hardware_mode,
            len(reranked_chunks),
            final_papers,
        )

        # Log performance insights
        if len(retrieved_chunks) > 0:
            efficiency = len(reranked_chunks) / len(retrieved_chunks) * 100
            logger.debug(
                "%s: Pipeline efficiency: %.1f%% (%d final / %d initial chunks) - %s",
                self.call_id,
                efficiency,
                len(reranked_chunks),
                len(retrieved_chunks),
                hardware_mode,
            )

        return reranked_chunks

    def format_answer(
        self,
        question: str,
        chunks: List[Any],
        llm: Any,
        articles: Dict[str, Any],
    ) -> str:
        """Generate the final answer text with source attributions and hardware info."""
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

        # Log final statistics with hardware info
        hardware_info = "GPU-accelerated" if self.has_gpu else "CPU-processed"
        logger.info(
            "%s: Generated answer using %d chunks from %d papers (%s)",
            self.call_id,
            len(chunks),
            len(titles),
            hardware_info,
        )

        # Add subtle hardware info to logs but not to user output
        logger.debug(
            "%s: Answer generation completed with %s processing",
            self.call_id,
            hardware_info,
        )

        return f"{answer}{srcs}"

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get current hardware configuration stats for monitoring."""
        return {
            "gpu_available": self.has_gpu,
            "hardware_mode": "GPU-accelerated" if self.has_gpu else "CPU-only",
            "call_id": self.call_id,
        }
