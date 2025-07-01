"""
Vectorstore class for managing document embeddings and retrieval using Milvus.
Implements singleton pattern for connection reuse across Streamlit sessions.
Enhanced with parallel PDF processing, batch embedding, and automatic GPU/CPU detection.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    utility,
)


# Import our GPU detection utility
from .gpu_detection import (
    detect_nvidia_gpu,
    get_optimal_index_config,
    log_index_configuration,
)
from .singleton_manager import VectorstoreSingleton

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class Vectorstore:
    """
    A class for managing document embeddings and retrieval using Milvus.
    Uses singleton pattern to reuse connections across instances.
    Enhanced with parallel PDF processing, batch embedding, and automatic GPU/CPU detection.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        metadata_fields: Optional[List[str]] = None,
        config: Any = None,
    ):
        """
        Initialize the document store with Milvus.

        Args:
            embedding_model: The embedding model to use
            metadata_fields: Fields to include in document metadata for filtering/retrieval
            config: Configuration object containing Milvus connection details
        """
        self.embedding_model = embedding_model
        self.config = config
        self.metadata_fields = metadata_fields or [
            "title",
            "paper_id",
            "page",
            "chunk_id",
        ]
        self.initialization_time = time.time()
        logger.info("Vectorstore initialized at: %s", self.initialization_time)

        # Track loaded papers to prevent duplicate loading
        self.loaded_papers = set()

        # Initialize Milvus connection parameters
        self.connection_args = {
            "host": config.milvus.host if config else "127.0.0.1",
            "port": config.milvus.port if config else 19530,
        }
        self.collection_name = (
            config.milvus.collection_name if config else "pdf_rag_documents"
        )
        self.db_name = config.milvus.db_name if config else "pdf_rag_db"

        # Get singleton instance
        self._singleton = VectorstoreSingleton()

        # GPU detection with config override (SINGLE CALL)
        self.has_gpu = detect_nvidia_gpu(config)

        # Additional check for force CPU mode
        if (
            config
            and hasattr(config, "gpu_detection")
            and getattr(config.gpu_detection, "force_cpu_mode", False)
        ):
            logger.info("🔧 Running in forced CPU mode (config override)")
            self.has_gpu = False

        # Configure index parameters AFTER determining GPU usage
        embedding_dim = config.milvus.embedding_dim if config else 768
        self.index_params, self.search_params = get_optimal_index_config(
            self.has_gpu, embedding_dim
        )

        # Log the configuration
        log_index_configuration(self.index_params, self.search_params)

        # Connect to Milvus (reuses existing connection if available)
        self._connect_milvus()
        self._ensure_collection_exists()

        # Initialize the LangChain Milvus vector store (reuses existing if available)
        self.vector_store = self._initialize_vector_store()
        # Store for document metadata (keeping for compatibility)
        self.documents: Dict[str, Document] = {}
        self.paper_metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Milvus vector store initialized with collection: %s (GPU: %s)",
            self.collection_name,
            "enabled" if self.has_gpu else "disabled",
        )

    def _connect_milvus(self) -> None:
        """Establish connection to Milvus server using singleton."""
        self._singleton.get_connection(
            self.connection_args["host"], self.connection_args["port"], self.db_name
        )

    def _initialize_vector_store(self) -> Milvus:
        """Initialize or load the Milvus vector store using singleton with GPU optimization."""
        self._ensure_collection_exists()  # Ensure collection exists before use

        # Create the base vector store
        vector_store = self._singleton.get_vector_store(
            self.collection_name, self.embedding_model, self.connection_args
        )

        # Configure search parameters based on hardware detection
        # This avoids passing search_params through LangChain methods
        if hasattr(vector_store, "_client") and self.has_gpu:
            logger.info("Configuring Milvus client for GPU-optimized search")
            # The GPU optimization will be handled at the collection level
            # through the index configuration, not search parameters

        return vector_store

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional filter dict for metadata
            **kwargs: Additional search parameters

        Returns:
            List of Document objects
        """
        # Convert filter dict to Milvus expression if provided
        expr = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                elif isinstance(value, list):
                    # For filtering by multiple values
                    values_str = ", ".join(
                        [f'"{v}"' if isinstance(v, str) else str(v) for v in value]
                    )
                    conditions.append(f"{key} in [{values_str}]")
                else:
                    conditions.append(f"{key} == {value}")
            expr = " and ".join(conditions) if conditions else None

        # Don't pass search_params to avoid conflicts with LangChain
        # The GPU optimization happens at the collection level
        results = self.vector_store.similarity_search(
            query=query, k=k, expr=expr, **kwargs
        )

        return results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Perform MMR search on the vector store.

        Args:
            query: Query string
            k: Number of results to return
            fetch_k: Number of results to fetch before MMR
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Optional filter dict for metadata
            **kwargs: Additional search parameters

        Returns:
            List of Document objects
        """
        # Convert filter dict to Milvus expression if provided
        expr = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                elif isinstance(value, list):
                    values_str = ", ".join(
                        [f'"{v}"' if isinstance(v, str) else str(v) for v in value]
                    )
                    conditions.append(f"{key} in [{values_str}]")
                else:
                    conditions.append(f"{key} == {value}")
            expr = " and ".join(conditions) if conditions else None

        # Don't pass search_params to avoid conflicts with LangChain
        # The GPU optimization happens at the collection level
        results = self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            expr=expr,
            **kwargs,
        )

        return results

    def _ensure_collection_exists(self):
        """Ensure the Milvus collection exists before trying to sync or add documents."""
        try:
            existing_collections = utility.list_collections()
            if self.collection_name not in existing_collections:
                logger.info(
                    "Collection %s does not exist. Creating schema...",
                    self.collection_name,
                )

                # Define schema
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        auto_id=False,
                        max_length=100,
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.config.milvus.embedding_dim if self.config else 768,
                    ),
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                    ),
                    FieldSchema(
                        name="paper_id",
                        dtype=DataType.VARCHAR,
                        max_length=100,
                    ),
                    FieldSchema(
                        name="title",
                        dtype=DataType.VARCHAR,
                        max_length=512,
                    ),
                    FieldSchema(
                        name="chunk_id",
                        dtype=DataType.INT64,
                    ),
                    FieldSchema(
                        name="page",
                        dtype=DataType.INT64,
                    ),
                    FieldSchema(
                        name="source",
                        dtype=DataType.VARCHAR,
                        max_length=512,
                    ),
                ]

                schema = CollectionSchema(
                    fields=fields,
                    description="RAG collection for embedded PDF chunks",
                    enable_dynamic_field=True,
                )

                # Create collection
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using="default",
                    shards_num=2,
                )
                logger.info("Created collection: %s", self.collection_name)

                # Create index on the embedding field with GPU/CPU optimization
                logger.info(
                    "Creating %s index on 'embedding' field for collection: %s",
                    self.index_params["index_type"],
                    self.collection_name,
                )

                self.collection.create_index(
                    field_name="embedding", index_params=self.index_params
                )

                index_type = self.index_params["index_type"]
                logger.info(
                    "Successfully created %s index on 'embedding' field for collection: %s",
                    index_type,
                    self.collection_name,
                )

            else:
                logger.info(
                    "Collection %s already exists. Loading it.", self.collection_name
                )
                self.collection = Collection(name=self.collection_name, using="default")

            self.collection.load()

            # Log collection statistics with GPU/CPU info
            num_entities = self.collection.num_entities
            gpu_info = " (GPU accelerated)" if self.has_gpu else " (CPU only)"
            logger.info(
                "Collection %s is loaded and ready with %d entities%s",
                self.collection_name,
                num_entities,
                gpu_info,
            )

        except Exception as e:
            logger.error("Failed to ensure collection exists: %s", e, exc_info=True)
            raise
