"""
NVIDIA NIM Reranker Utility for Milvus Integration
"""

import logging
import os
from typing import Any, List

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


def rank_papers_by_query(
    vector_store, query: str, config: Any, paper_ids: List[str], top_k: int = 5
) -> List[str]:
    """
    Rank papers by relevance to the query using NVIDIA's off-the-shelf re-ranker.

    This function retrieves chunks for specified papers from Milvus, aggregates them
    by paper, ranks them using the NVIDIA model, and returns the top-k papers.

    Args:
        vector_store: The Milvus vector store instance
        query (str): The query string.
        config (Any): Configuration containing reranker settings (model, api_key).
        paper_ids (List[str]): List of paper IDs to rank
        top_k (int): Number of top papers to return.

    Returns:
        List of paper_ids sorted by relevance.
    """
    logger.info(
        "Starting NVIDIA re-ranker for query: '%s' with %d papers, top_k=%d",
        query[:50] + "..." if len(query) > 50 else query,
        len(paper_ids),
        top_k,
    )

    # If we have fewer papers than top_k, just return all
    if len(paper_ids) <= top_k:
        logger.info(
            "Number of papers (%d) <= top_k (%d), returning all papers",
            len(paper_ids),
            top_k,
        )
        return paper_ids

    try:
        # Retrieve all chunks for the specified papers from Milvus
        # We'll get a reasonable number of chunks per paper for aggregation
        chunks_per_paper = 10  # Adjust based on your needs
        total_chunks_needed = len(paper_ids) * chunks_per_paper

        logger.info(
            "Retrieving up to %d chunks for %d papers from Milvus",
            total_chunks_needed,
            len(paper_ids),
        )

        # Query Milvus for chunks from these specific papers
        filter_dict = {"paper_id": paper_ids}
        chunks = vector_store.similarity_search(
            query=query,  # Use the query to get relevant chunks
            k=total_chunks_needed,
            filter=filter_dict,
        )

        logger.info("Retrieved %d chunks from Milvus", len(chunks))

        # Aggregate chunks by paper
        paper_texts = {}
        for chunk in chunks:
            paper_id = chunk.metadata.get("paper_id")
            if paper_id and paper_id in paper_ids:
                if paper_id not in paper_texts:
                    paper_texts[paper_id] = []
                paper_texts[paper_id].append(chunk.page_content)

        # Create aggregated documents for reranking
        aggregated_documents = []
        for paper_id, texts in paper_texts.items():
            aggregated_text = " ".join(texts)
            # Truncate if too long (NVIDIA reranker has token limits)
            max_chars = 8000  # Adjust based on model limits
            if len(aggregated_text) > max_chars:
                aggregated_text = aggregated_text[:max_chars] + "..."

            aggregated_documents.append(
                Document(page_content=aggregated_text, metadata={"paper_id": paper_id})
            )

        logger.info(
            "Aggregated %d papers into documents for reranking",
            len(aggregated_documents),
        )

        # Ensure we have documents to rerank
        if not aggregated_documents:
            logger.warning("No documents to rerank, returning original paper order")
            return paper_ids[:top_k]

        # Instantiate the NVIDIA re-ranker client using provided config
        api_key = config.reranker.api_key
        if not api_key:
            logger.error("No NVIDIA API key found in configuration for reranking")
            raise ValueError(
                "Configuration 'reranker.api_key' must be set for reranking"
            )

        logger.info("Using NVIDIA reranker model: %s", config.reranker.model)

        # Initialize reranker with truncation to handle long documents
        reranker = NVIDIARerank(
            model=config.reranker.model,
            api_key=api_key,
            truncate="END",  # Truncate at the end if too long
        )

        # Get the ranked list of documents based on the query
        logger.info("Calling NVIDIA reranker API...")
        response = reranker.compress_documents(
            query=query, documents=aggregated_documents
        )

        logger.info("Received %d documents from NVIDIA reranker", len(response))

        # Extract paper IDs in ranked order
        ranked_papers = []
        for doc in response:
            paper_id = doc.metadata.get("paper_id")
            if paper_id and paper_id not in ranked_papers:
                ranked_papers.append(paper_id)
                if len(ranked_papers) >= top_k:
                    break

        # Add any missing papers that weren't in the reranker results
        for paper_id in paper_ids:
            if paper_id not in ranked_papers and len(ranked_papers) < top_k:
                ranked_papers.append(paper_id)

        logger.info(
            "Top %d papers after reranking: %s", len(ranked_papers), ranked_papers
        )
        return ranked_papers

    except Exception as e:
        logger.error("NVIDIA reranker failed: %s", e)
        logger.info("Falling back to original paper order")
        # Return the first top_k papers as fallback
        return paper_ids[:top_k]
