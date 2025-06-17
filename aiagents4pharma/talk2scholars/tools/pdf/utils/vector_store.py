"""
Vectorstore class for managing document embeddings and retrieval using Milvus.
"""

import logging
import os
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


class Vectorstore:
    """
    A class for managing document embeddings and retrieval using Milvus.
    Provides unified access to documents across multiple papers.
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

        # Connect to Milvus
        self._connect_milvus()

        # Initialize the LangChain Milvus vector store
        self.vector_store = self._initialize_vector_store()

        # Store for document metadata (keeping for compatibility)
        self.documents: Dict[str, Document] = {}
        self.paper_metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Milvus vector store initialized with collection: %s", self.collection_name
        )

    def _connect_milvus(self) -> None:
        """Establish connection to Milvus server."""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.connection_args["host"],
                port=self.connection_args["port"],
            )
            logger.info(
                "Connected to Milvus at %s:%s",
                self.connection_args["host"],
                self.connection_args["port"],
            )

            # Check if database exists, create if not
            from pymilvus import db

            existing_dbs = db.list_database()
            if self.db_name not in existing_dbs:
                db.create_database(self.db_name)
                logger.info("Created database: %s", self.db_name)

            # Use the database
            db.using_database(self.db_name)
            logger.info("Using database: %s", self.db_name)

        except MilvusException as e:
            logger.error("Failed to connect to Milvus: %s", e)
            raise

    def _initialize_vector_store(self) -> Milvus:
        """Initialize or load the Milvus vector store."""
        try:
            # Create LangChain Milvus instance
            vector_store = Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                # Define text field for storing document content
                text_field="text",
                # Auto-create collection with proper schema
                auto_id=False,  # We'll provide our own IDs
                drop_old=False,  # Don't drop existing collection
                consistency_level="Strong",
            )

            logger.info("Milvus vector store initialized/loaded successfully")
            return vector_store

        except Exception as e:
            logger.error("Failed to initialize Milvus vector store: %s", e)
            raise

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
