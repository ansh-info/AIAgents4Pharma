"""
Vectorstore class for managing PDF embeddings with Milvus.
Manages GPU normalization and similarity search and MMR operations.
With automatic handling of COSINE to IP conversion for GPU compatibility.
Supports both GPU and CPU configurations.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .collection_manager import ensure_collection_exists
from .gpu_detection import (
    detect_nvidia_gpu,
    get_optimal_index_config,
    log_index_configuration,
)
from .singleton_manager import VectorstoreSingleton
from .vector_normalization import wrap_embedding_model_if_needed

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class Vectorstore:
    """
    Enhanced Vectorstore class with GPU normalization support.
    Automatically handles COSINE -> IP conversion for GPU compatibility.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        metadata_fields: Optional[List[str]] = None,
        config: Any = None,
    ):
        """
        Initialize the document store with Milvus and GPU optimization.

        Args:
            embedding_model: The embedding model to use
            metadata_fields: Fields to include in document metadata
            config: Configuration object containing Milvus connection details
        """
        self.config = config
        self.metadata_fields = metadata_fields or [
            "title",
            "paper_id",
            "page",
            "chunk_id",
        ]
        self.initialization_time = time.time()

        # GPU detection with config override (SINGLE CALL)
        self.has_gpu = detect_nvidia_gpu(config)

        # Additional check for force CPU mode
        if (
            config
            and hasattr(config, "gpu_detection")
            and getattr(config.gpu_detection, "force_cpu_mode", False)
        ):
            logger.info("Running in forced CPU mode (config override)")
            self.has_gpu = False

        # Determine if we want to use COSINE similarity
        self.use_cosine = True  # Default preference
        if config and hasattr(config, "similarity_metric"):
            self.use_cosine = getattr(config.similarity_metric, "use_cosine", True)

        # Wrap embedding model with normalization if needed for GPU
        self.original_embedding_model = embedding_model
        self.embedding_model = wrap_embedding_model_if_needed(
            embedding_model, self.has_gpu, self.use_cosine
        )

        # Configure index parameters AFTER determining GPU usage and normalization
        embedding_dim = config.milvus.embedding_dim if config else 768
        self.index_params, self.search_params = get_optimal_index_config(
            self.has_gpu, embedding_dim, self.use_cosine
        )

        # Log the configuration
        log_index_configuration(self.index_params, self.search_params, self.use_cosine)

        # Track loaded papers to prevent duplicate loading
        self.loaded_papers = set()

        # Initialize Milvus connection parameters with environment variable fallback
        self.connection_args = {
            "host": (
                config.milvus.host if config else os.getenv("MILVUS_HOST", "127.0.0.1")
            ),
            "port": (
                config.milvus.port if config else int(os.getenv("MILVUS_PORT", "19530"))
            ),
        }
        # Log the connection parameters being used
        logger.info(
            "Using Milvus connection: %s:%s",
            self.connection_args["host"],
            self.connection_args["port"],
        )
        self.collection_name = (
            config.milvus.collection_name if config else "pdf_rag_documents"
        )
        self.db_name = config.milvus.db_name if config else "pdf_rag_db"

        # Get singleton instance
        self._singleton = VectorstoreSingleton()

        # Connect to Milvus (reuses existing connection if available)
        self._connect_milvus()

        # Create collection with proper metric type
        self.collection = ensure_collection_exists(
            self.collection_name, self.config, self.index_params, self.has_gpu
        )

        # Collection is now the primary interface (no LangChain wrapper needed)

        # Load existing papers AFTER vector store is ready
        self._load_existing_paper_ids()

        # CRITICAL: Load collection into memory/GPU after any existing data is identified
        logger.info(
            "Calling _ensure_collection_loaded() for %s processing...",
            "GPU" if self.has_gpu else "CPU",
        )
        self._ensure_collection_loaded()

        # Store for document metadata (keeping for compatibility)
        self.documents: Dict[str, Document] = {}
        self.paper_metadata: Dict[str, Dict[str, Any]] = {}

        # Log final configuration
        metric_info = (
            "IP (normalized for COSINE)"
            if self.has_gpu and self.use_cosine
            else self.index_params["metric_type"]
        )

        logger.info(
            "Milvus vector store initialized with collection: %s (GPU: %s, Metric: %s)",
            self.collection_name,
            "enabled" if self.has_gpu else "disabled",
            metric_info,
        )

    def _connect_milvus(self) -> None:
        """Establish connection to Milvus server using singleton."""
        self._singleton.get_connection(
            self.connection_args["host"], self.connection_args["port"], self.db_name
        )

    # Removed _initialize_vector_store - no longer needed with pure PyMilvus

    def _load_existing_paper_ids(self):
        """Load already embedded paper IDs using pure PyMilvus collection."""
        logger.info("Checking for existing papers in PyMilvus collection...")

        # Force flush and check entity count
        self.collection.flush()
        num_entities = self.collection.num_entities

        logger.info("PyMilvus collection entity count: %d", num_entities)

        if num_entities > 0:
            logger.info("Loading existing paper IDs from PyMilvus collection...")

            results = self.collection.query(
                expr="",  # No filter - get all
                output_fields=["paper_id"],
                limit=16384,  # Max limit
                consistency_level="Strong",
            )

            # Extract unique paper IDs
            existing_paper_ids = set(result["paper_id"] for result in results)
            self.loaded_papers.update(existing_paper_ids)

            logger.info("Found %d unique papers in collection", len(existing_paper_ids))
        else:
            logger.info("Collection is empty - no existing papers")

    # Removed custom similarity_search, similarity_search_with_score, and max_marginal_relevance_search methods
    # These were causing type checking issues with PyMilvus SearchFuture/SearchResult objects
    # The system now uses direct PyMilvus collection.search() calls where needed

    def _ensure_collection_loaded(self):
        """Ensure collection is loaded into memory/GPU after data insertion."""
        # Use direct PyMilvus collection reference
        collection = self.collection

        # Force flush to ensure we see all data
        logger.info("Flushing collection to ensure data visibility...")
        collection.flush()

        # Check entity count after flush
        num_entities = collection.num_entities
        logger.info("Collection entity count after flush: %d", num_entities)

        if num_entities > 0:
            hardware_type = "GPU" if self.has_gpu else "CPU"
            logger.info(
                "Loading collection with %d entities into %s memory...",
                num_entities,
                hardware_type,
            )

            # Load collection into memory (CPU or GPU)
            collection.load()

            # Verify loading was successful
            final_count = collection.num_entities
            logger.info(
                "Collection successfully loaded into %s memory with %d entities",
                hardware_type,
                final_count,
            )
        else:
            logger.info("Collection is empty, skipping load operation")

    def add_documents(self, documents: List[Document], ids: List[str]) -> None:
        """
        Add documents to the collection using pure PyMilvus.
        Handles embedding generation and insertion.
        """
        if not documents or not ids:
            logger.warning("No documents or IDs provided for insertion")
            return

        if len(documents) != len(ids):
            raise ValueError("Number of documents must match number of IDs")

        # Extract texts and generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)

        # Prepare data for insertion
        entities = []
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            entity = {
                "id": doc_id,
                "text": doc.page_content,
                "embedding": embeddings[i],
                "paper_id": doc.metadata.get("paper_id", ""),
                "title": doc.metadata.get("title", ""),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
            }
            entities.append(entity)

        # Insert into collection
        self.collection.insert(entities)
        logger.info("Inserted %d documents into collection", len(documents))

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding configuration."""
        return {
            "has_gpu": self.has_gpu,
            "use_cosine": self.use_cosine,
            "metric_type": self.index_params["metric_type"],
            "index_type": self.index_params["index_type"],
            "normalization_enabled": hasattr(self.embedding_model, "normalize_for_gpu"),
            "original_model_type": type(self.original_embedding_model).__name__,
            "wrapped_model_type": type(self.embedding_model).__name__,
        }
