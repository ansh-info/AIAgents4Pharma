"""
Retrieve relevant chunks from a Milvus vector store using MMR (Maximal Marginal Relevance).
"""

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document


# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def retrieve_relevant_chunks(
    vector_store,
    query: str,
    paper_ids: Optional[List[str]] = None,
    top_k: int = 25,
    mmr_diversity: float = 1.00,
) -> List[Document]:
    """
    Retrieve the most relevant chunks for a query using maximal marginal relevance.

    Args:
        vector_store: The Milvus vector store instance
        query: Query string
        paper_ids: Optional list of paper IDs to filter by
        top_k: Number of chunks to retrieve
        mmr_diversity: Diversity parameter for MMR (0=max diversity, 1=max relevance)

    Returns:
        List of document chunks
    """
    if not vector_store:
        logger.error("Vector store is not initialized")
        return []

    # Prepare filter for paper_ids if provided
    filter_dict = None
    if paper_ids:
        logger.info("Filtering retrieval to papers: %s", paper_ids)
        filter_dict = {"paper_id": paper_ids}

    try:
        # Use Milvus's built-in MMR search
        logger.info(
            "Performing MMR search with query: '%s', k=%d, diversity=%.2f",
            query[:50] + "..." if len(query) > 50 else query,
            top_k,
            mmr_diversity,
        )

        # Perform MMR search using the Milvus vector store
        results = vector_store.max_marginal_relevance_search(
            query=query,
            k=top_k,
            fetch_k=top_k * 4,  # Fetch more candidates for better MMR results
            lambda_mult=mmr_diversity,
            filter=filter_dict,
        )

        logger.info("Retrieved %d chunks using MMR from Milvus", len(results))

        # Log some details about retrieved chunks for debugging
        if results and logger.isEnabledFor(logging.DEBUG):
            paper_counts = {}
            for doc in results:
                paper_id = doc.metadata.get("paper_id", "unknown")
                paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
            logger.debug("Chunks per paper: %s", paper_counts)

        return results

    except Exception as e:
        logger.error("Error during MMR search: %s", e)
        # Fallback to regular similarity search if MMR fails
        try:
            logger.info("Falling back to regular similarity search")
            results = vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=filter_dict,
            )
            logger.info("Retrieved %d chunks using similarity search", len(results))
            return results
        except Exception as fallback_error:
            logger.error("Fallback search also failed: %s", fallback_error)
            return []
