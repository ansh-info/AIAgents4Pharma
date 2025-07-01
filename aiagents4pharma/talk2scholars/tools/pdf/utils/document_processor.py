"""
Document processing utilities for loading and splitting PDFs.
"""

import logging
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_and_split_pdf(
    paper_id: str,
    pdf_url: str,
    paper_metadata: Dict[str, Any],
    config: Any,
    metadata_fields: List[str],
    documents_dict: Dict[str, Document],
) -> List[Document]:
    """
    Load a PDF and split it into chunks.

    Args:
        paper_id: Unique identifier for the paper
        pdf_url: URL to the PDF
        paper_metadata: Metadata about the paper

    Returns:
        List of document chunks with metadata
    """
    logger.info("Loading PDF for paper %s from %s", paper_id, pdf_url)

    # Load the PDF
    loader = PyPDFLoader(pdf_url)
    documents = loader.load()
    logger.info("Loaded %d pages from paper %s", len(documents), paper_id)

    # Create text splitter according to provided configuration
    if config is None:
        raise ValueError("Configuration is required for text splitting in Vectorstore.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split documents
    chunks = splitter.split_documents(documents)
    logger.info("Split paper %s into %d chunks", paper_id, len(chunks))

    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        # Create unique ID for each chunk
        chunk_id = f"{paper_id}_{i}"

        # Enhance metadata
        chunk.metadata.update(
            {
                "paper_id": paper_id,
                "title": paper_metadata.get("Title", "Unknown"),
                "chunk_id": i,
                "page": chunk.metadata.get("page", 0),
                "source": pdf_url,
            }
        )

        # Add any additional metadata fields
        for field in metadata_fields:
            if field in paper_metadata and field not in chunk.metadata:
                chunk.metadata[field] = paper_metadata[field]

        # Store in local dict for compatibility
        documents_dict[chunk_id] = chunk

    return chunks
