"""
Helper class for PDF Q&A tool orchestration: state validation, vectorstore init,
paper loading, reranking, and answer formatting.
"""

import logging
from typing import Any, Dict, List, Optional

from .generate_answer import generate_answer
from .nvidia_nim_reranker import rank_papers_by_query
from .retrieve_chunks import retrieve_relevant_chunks
from .vector_store import Vectorstore

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
        # Check if we already have a vector store initialized
        if self.vector_store is not None:
            logger.info("%s: Using existing Milvus vector store", self.call_id)
            # Get current stats
            stats = self.vector_store.get_collection_stats()
            logger.info(
                "%s: Vector store stats - Papers: %d, Entities: %s",
                self.call_id,
                stats.get("num_loaded_papers", 0),
                stats.get("num_entities", "unknown"),
            )
            return self.vector_store

        # Create new vector store with Milvus
        logger.info("%s: Initializing new Milvus vector store", self.call_id)
        self.vector_store = Vectorstore(embedding_model=emb_model, config=self.config)
        return self.vector_store

    def load_candidate_papers(
        self,
        vs: Vectorstore,
        articles: Dict[str, Any],
        candidates: List[str],
    ) -> None:
        """Ensure each candidate paper is loaded into the Milvus vector store."""
        papers_to_load = []

        # Check which papers need to be loaded
        for pid in candidates:
            if pid not in vs.loaded_papers:
                pdf_url = articles.get(pid, {}).get("pdf_url")
                if pdf_url:
                    papers_to_load.append(pid)
                else:
                    logger.warning(
                        "%s: No PDF URL found for paper %s", self.call_id, pid
                    )

        if not papers_to_load:
            logger.info(
                "%s: All %d candidate papers already loaded in Milvus",
                self.call_id,
                len(candidates),
            )
            return

        logger.info(
            "%s: Loading %d new papers into Milvus (already loaded: %d)",
            self.call_id,
            len(papers_to_load),
            len(candidates) - len(papers_to_load),
        )

        # Load each paper
        for pid in papers_to_load:
            pdf_url = articles[pid]["pdf_url"]
            try:
                vs.add_paper(pid, pdf_url, articles[pid])
                logger.info("%s: Successfully loaded paper %s", self.call_id, pid)
            except (IOError, ValueError) as exc:
                logger.warning("%s: Error loading paper %s: %s", self.call_id, pid, exc)

        # No need to call build_vector_store() with Milvus as it's built incrementally

    def run_reranker(
        self,
        vs: Vectorstore,
        query: str,
        candidates: List[str],
    ) -> List[str]:
        """Rank papers by relevance and return filtered paper IDs."""
        try:
            # Filter candidates to only include loaded papers
            loaded_candidates = [pid for pid in candidates if pid in vs.loaded_papers]

            if not loaded_candidates:
                logger.warning(
                    "%s: No candidates are loaded in vector store", self.call_id
                )
                return []

            if len(loaded_candidates) < len(candidates):
                logger.info(
                    "%s: Reranking %d loaded papers out of %d candidates",
                    self.call_id,
                    len(loaded_candidates),
                    len(candidates),
                )

            # Call the updated reranker function with paper_ids
            ranked = rank_papers_by_query(
                vs,
                query,
                self.config,
                paper_ids=loaded_candidates,
                top_k=self.config.top_k_papers,
            )

            logger.info("%s: Papers after NVIDIA reranking: %s", self.call_id, ranked)
            return ranked

        except (ValueError, RuntimeError) as exc:
            logger.error("%s: NVIDIA reranker failed: %s", self.call_id, exc)
            logger.info(
                "%s: Falling back to first %d loaded papers",
                self.call_id,
                self.config.top_k_papers,
            )
            # Return first k loaded papers as fallback
            loaded = [pid for pid in candidates if pid in vs.loaded_papers]
            return loaded[: self.config.top_k_papers]

    def retrieve_chunks(
        self,
        vs: Vectorstore,
        query: str,
        paper_ids: List[str],
    ) -> List[Any]:
        """Retrieve relevant chunks using the updated retrieve function."""
        # Call the updated retrieve_relevant_chunks function
        chunks = retrieve_relevant_chunks(
            vs,
            query=query,
            paper_ids=paper_ids,
            top_k=self.config.top_k_chunks,
            mmr_diversity=1.0,  # Can be made configurable if needed
        )
        return chunks

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
