"""
Batch processing utilities for adding multiple papers to vector store.
"""

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Set, Tuple

from langchain_core.documents import Document

from .document_processor import load_and_split_pdf

logger = logging.getLogger(__name__)


def add_papers_batch(
    papers_to_add: List[Tuple[str, str, Dict[str, Any]]],
    vector_store: Any,  # The LangChain Milvus vector store
    loaded_papers: Set[str],  # Set to track loaded papers
    paper_metadata: Dict[str, Dict[str, Any]],  # Dict to store paper metadata
    documents: Dict[str, Document],  # Dict to store document chunks
    config: Any,
    metadata_fields: List[str],
    has_gpu: bool,
    max_workers: int = 5,
    batch_size: int = 100,
) -> None:
    """
    Add multiple papers to the document store in parallel with batch embedding.

    Args:
        papers_to_add: List of tuples (paper_id, pdf_url, paper_metadata)
        vector_store: The LangChain Milvus vector store instance
        loaded_papers: Set to track which papers are already loaded
        paper_metadata: Dictionary to store paper metadata
        documents: Dictionary to store document chunks
        config: Configuration object
        metadata_fields: List of metadata fields to include
        has_gpu: Whether GPU is available for processing
        max_workers: Maximum number of parallel PDF loading workers
        batch_size: Number of chunks to embed in a single batch
    """
    if not papers_to_add:
        logger.info("No papers to add")
        return

    # Filter out already loaded papers BEFORE processing
    papers_to_process = []
    for paper_id, pdf_url, metadata in papers_to_add:
        if paper_id in loaded_papers:
            logger.debug("Paper %s already loaded, skipping", paper_id)
        else:
            papers_to_process.append((paper_id, pdf_url, metadata))

    if not papers_to_process:
        logger.info(
            "Skipping %d already-loaded papers",
            len(papers_to_add) - len(papers_to_process),
        )
        logger.info("All %d papers are already loaded", len(papers_to_add))
        return

    # Log GPU status for this batch
    gpu_status = "GPU acceleration" if has_gpu else "CPU processing"
    logger.info(
        "Starting PARALLEL batch processing of %d papers with %d workers (%s)",
        len(papers_to_process),
        max_workers,
        gpu_status,
    )
    start_time = time.time()

    # Step 1: Load and split PDFs in parallel
    all_chunks = []
    all_ids = []
    successful_papers = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all PDF loading tasks AT ONCE
        future_to_paper = {
            executor.submit(
                load_and_split_pdf,
                paper_id,
                pdf_url,
                metadata,
                config,
                metadata_fields,
                documents,
            ): (paper_id, metadata)
            for paper_id, pdf_url, metadata in papers_to_process
        }

        logger.info(
            "Submitted %d PDF loading tasks to thread pool", len(future_to_paper)
        )

        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_paper):
            paper_id, metadata = future_to_paper[future]
            completed += 1
            try:
                chunks = future.result()

                # Generate IDs for these chunks
                chunk_ids = [f"{paper_id}_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_ids.extend(chunk_ids)
                successful_papers.append(paper_id)

                # Store paper metadata
                paper_metadata[paper_id] = metadata

                logger.info(
                    "Progress: %d/%d - Loaded paper %s (%d chunks)",
                    completed,
                    len(papers_to_process),
                    paper_id,
                    len(chunks),
                )

            except Exception as e:
                logger.error(
                    "Progress: %d/%d - Failed to load paper %s: %s",
                    completed,
                    len(papers_to_process),
                    paper_id,
                    e,
                )

    load_time = time.time() - start_time
    logger.info(
        "PARALLEL LOADING COMPLETE: Processed %d/%d papers into %d chunks in %.2f seconds (%.2f papers/sec)",
        len(successful_papers),
        len(papers_to_process),
        len(all_chunks),
        load_time,
        len(successful_papers) / load_time if load_time > 0 else 0,
    )

    if not all_chunks:
        logger.warning("No chunks to add to vector store")
        return

    # Step 2: Add chunks to Milvus in batches
    try:
        embed_start = time.time()
        total_chunks = len(all_chunks)

        logger.info(
            "Starting BATCH EMBEDDING of %d chunks in batches of %d (%s)",
            total_chunks,
            batch_size,
            gpu_status,
        )

        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = all_chunks[i:batch_end]
            batch_ids = all_ids[i:batch_end]

            logger.info(
                "Processing embedding batch %d/%d (chunks %d-%d of %d) - %s",
                (i // batch_size) + 1,
                (total_chunks + batch_size - 1) // batch_size,
                i + 1,
                batch_end,
                total_chunks,
                gpu_status,
            )

            # Extract texts for embedding
            texts = [chunk.page_content for chunk in batch_chunks]

            # Log embedding API call
            logger.info(
                "Calling embedding API for batch of %d texts (avg length: %d chars) - %s",
                len(texts),
                sum(len(t) for t in texts) // len(texts) if texts else 0,
                gpu_status,
            )

            # Add to Milvus (this will trigger embedding and use GPU/CPU index)
            vector_store.add_documents(
                documents=batch_chunks,
                ids=batch_ids,
            )

            def verify_insert_success(vector_store, batch_size, batch_num):
                """Verify that documents were actually inserted."""
                try:
                    # Get the underlying Milvus collection
                    collection = (
                        vector_store.col
                    )  # LangChain Milvus stores collection in .col

                    # Force flush to ensure data is persisted
                    collection.flush()

                    # Check entity count after flush
                    entity_count = collection.num_entities
                    logger.info(
                        "POST-INSERT verification batch %d: Collection now has %d entities",
                        batch_num,
                        entity_count,
                    )

                    # If we have entities, sample a few to verify structure
                    if entity_count > 0:
                        sample_results = collection.query(
                            expr="", output_fields=["paper_id"], limit=3
                        )
                        sample_papers = [
                            r.get("paper_id", "unknown") for r in sample_results
                        ]
                        logger.info("Sample paper IDs in collection: %s", sample_papers)

                except Exception as e:
                    logger.error("Insert verification failed: %s", e)

            # Add this call in batch_processor.py after:
            # vector_store.add_documents(documents=batch_chunks, ids=batch_ids)

            verify_insert_success(
                vector_store, len(batch_chunks), (i // batch_size) + 1
            )

            logger.info(
                "Successfully embedded and stored batch %d/%d with %s",
                (i // batch_size) + 1,
                (total_chunks + batch_size - 1) // batch_size,
                gpu_status,
            )

        embed_time = time.time() - embed_start
        logger.info(
            "BATCH EMBEDDING COMPLETE: Embedded %d chunks in %.2f seconds (%.2f chunks/sec) - %s",
            total_chunks,
            embed_time,
            total_chunks / embed_time if embed_time > 0 else 0,
            gpu_status,
        )

        # Update loaded papers
        for paper_id in successful_papers:
            loaded_papers.add(paper_id)

        total_time = time.time() - start_time
        logger.info(
            "FULL BATCH PROCESSING COMPLETE: %d papers, %d chunks in %.2f seconds total (%.2f sec/paper) - %s",
            len(successful_papers),
            total_chunks,
            total_time,
            total_time / len(successful_papers) if successful_papers else 0,
            gpu_status,
        )

    except Exception as e:
        logger.error("Failed to add chunks to Milvus: %s", e)
        raise
