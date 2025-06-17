"""
Vectorstore class for managing document embeddings and retrieval using Milvus.
Implements singleton pattern for connection reuse across Streamlit sessions.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility, Collection
from pymilvus.exceptions import MilvusException


# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))


class VectorstoreSingleton:
    """Singleton manager for Milvus connections and vector stores."""

    _instance = None
    _lock = threading.Lock()
    _connections = {}  # Store connections by connection string
    _vector_stores = {}  # Store vector stores by collection name
    _event_loops = {}  # Store event loops by thread ID

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for current thread."""
        thread_id = threading.get_ident()

        if thread_id not in self._event_loops:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self._event_loops[thread_id] = loop
            logger.info("Created new event loop for thread %s", thread_id)

        return self._event_loops[thread_id]

    def get_connection(self, host: str, port: int, db_name: str) -> str:
        """Get or create a Milvus connection."""
        conn_key = f"{host}:{port}/{db_name}"

        if conn_key not in self._connections:
            try:
                # Check if already connected
                if connections.has_connection("default"):
                    connections.remove_connection("default")

                # Connect to Milvus
                connections.connect(
                    alias="default",
                    host=host,
                    port=port,
                )
                logger.info("Connected to Milvus at %s:%s", host, port)

                # Check if database exists, create if not
                from pymilvus import db

                existing_dbs = db.list_database()
                if db_name not in existing_dbs:
                    db.create_database(db_name)
                    logger.info("Created database: %s", db_name)

                # Use the database
                db.using_database(db_name)
                logger.info("Using database: %s", db_name)

                self._connections[conn_key] = "default"

            except MilvusException as e:
                logger.error("Failed to connect to Milvus: %s", e)
                raise

        return self._connections[conn_key]

    def get_vector_store(
        self,
        collection_name: str,
        embedding_model: Embeddings,
        connection_args: Dict[str, Any],
    ) -> Milvus:
        """Get or create a vector store for a collection."""
        if collection_name not in self._vector_stores:
            # Ensure event loop exists for this thread
            self.get_event_loop()

            # Create LangChain Milvus instance
            vector_store = Milvus(
                embedding_function=embedding_model,
                collection_name=collection_name,
                connection_args=connection_args,
                text_field="text",
                auto_id=False,
                drop_old=False,
                consistency_level="Strong",
            )

            self._vector_stores[collection_name] = vector_store
            logger.info("Created new vector store for collection: %s", collection_name)

        return self._vector_stores[collection_name]


class Vectorstore:
    """
    A class for managing document embeddings and retrieval using Milvus.
    Uses singleton pattern to reuse connections across instances.
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

        # Connect to Milvus (reuses existing connection if available)
        self._connect_milvus()

        # Initialize the LangChain Milvus vector store (reuses existing if available)
        self.vector_store = self._initialize_vector_store()

        # Store for document metadata (keeping for compatibility)
        self.documents: Dict[str, Document] = {}
        self.paper_metadata: Dict[str, Dict[str, Any]] = {}

        # Sync loaded papers from existing collection
        self._sync_loaded_papers()

        logger.info(
            "Milvus vector store initialized with collection: %s", self.collection_name
        )

    def _connect_milvus(self) -> None:
        """Establish connection to Milvus server using singleton."""
        self._singleton.get_connection(
            self.connection_args["host"], self.connection_args["port"], self.db_name
        )

    def _initialize_vector_store(self) -> Milvus:
        """Initialize or load the Milvus vector store using singleton."""
        return self._singleton.get_vector_store(
            self.collection_name, self.embedding_model, self.connection_args
        )

    def _sync_loaded_papers(self) -> None:
        """Sync loaded papers from existing collection."""
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)

                # Load collection to memory if not already loaded
                collection.load()

                # Query to get unique paper_ids
                # Note: This is a simplified approach. For large collections,
                # you might want to implement pagination or use a more efficient method
                if collection.num_entities > 0:
                    # Query a sample to get paper IDs
                    results = collection.query(
                        expr="paper_id != ''",
                        output_fields=["paper_id"],
                        limit=10000,  # Adjust based on your needs
                    )

                    # Extract unique paper IDs
                    paper_ids = set()
                    for result in results:
                        if "paper_id" in result:
                            paper_ids.add(result["paper_id"])

                    self.loaded_papers = paper_ids
                    logger.info(
                        "Synced %d loaded papers from existing collection",
                        len(self.loaded_papers),
                    )
        except Exception as e:
            logger.warning("Could not sync loaded papers: %s", e)

    def add_paper(
        self,
        paper_id: str,
        pdf_url: str,
        paper_metadata: Dict[str, Any],
    ) -> None:
        """
        Add a paper to the document store.

        Args:
            paper_id: Unique identifier for the paper
            pdf_url: URL to the PDF
            paper_metadata: Metadata about the paper
        """
        # Skip if already loaded
        if paper_id in self.loaded_papers:
            logger.info("Paper %s already loaded, skipping", paper_id)
            return

        logger.info("Loading paper %s from %s", paper_id, pdf_url)

        # Store paper metadata
        self.paper_metadata[paper_id] = paper_metadata

        # Load the PDF and split into chunks according to Hydra config
        loader = PyPDFLoader(pdf_url)
        documents = loader.load()
        logger.info("Loaded %d pages from %s", len(documents), paper_id)

        # Create text splitter according to provided configuration
        if self.config is None:
            raise ValueError(
                "Configuration is required for text splitting in Vectorstore."
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Split documents and add metadata for each chunk
        chunks = splitter.split_documents(documents)
        logger.info("Split %s into %d chunks", paper_id, len(chunks))

        # Prepare documents for Milvus with enhanced metadata
        milvus_docs = []
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
            for field in self.metadata_fields:
                if field in paper_metadata and field not in chunk.metadata:
                    chunk.metadata[field] = paper_metadata[field]

            # Store in local dict for compatibility
            self.documents[chunk_id] = chunk
            milvus_docs.append(chunk)

        # Add documents to Milvus vector store
        try:
            # Generate IDs for documents
            ids = [f"{paper_id}_{i}" for i in range(len(milvus_docs))]

            # Add to Milvus
            self.vector_store.add_documents(
                documents=milvus_docs,
                ids=ids,
            )

            logger.info(
                "Added %d chunks from paper %s to Milvus", len(chunks), paper_id
            )

            # Mark as loaded
            self.loaded_papers.add(paper_id)

        except Exception as e:
            logger.error("Failed to add paper %s to Milvus: %s", paper_id, e)
            raise

    def build_vector_store(self) -> None:
        """
        For compatibility with existing code.
        With Milvus, the vector store is built incrementally as documents are added.
        """
        if not self.documents:
            logger.warning("No documents added to build vector store")
            return

        logger.info(
            "Vector store already built with %d documents in Milvus",
            len(self.documents),
        )

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

        # Perform search
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

        # Perform MMR search
        results = self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            expr=expr,
            **kwargs,
        )

        return results

    def delete_collection(self) -> None:
        """Delete the entire collection. Use with caution!"""
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.drop()
                logger.info("Dropped collection: %s", self.collection_name)

                # Clear from singleton cache
                if self.collection_name in self._singleton._vector_stores:
                    del self._singleton._vector_stores[self.collection_name]

                # Clear loaded papers
                self.loaded_papers.clear()
            else:
                logger.info("Collection %s does not exist", self.collection_name)
        except MilvusException as e:
            logger.error("Failed to drop collection: %s", e)
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                stats = {
                    "name": self.collection_name,
                    "num_entities": collection.num_entities,
                    "loaded_papers": list(self.loaded_papers),
                    "num_loaded_papers": len(self.loaded_papers),
                }
                return stats
            else:
                return {"error": f"Collection {self.collection_name} does not exist"}
        except MilvusException as e:
            logger.error("Failed to get collection stats: %s", e)
            return {"error": str(e)}
