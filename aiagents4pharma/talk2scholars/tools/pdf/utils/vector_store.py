"""
Vectorstore class for managing document embeddings and retrieval using Milvus.
Implements singleton pattern for connection reuse across Streamlit sessions.
Enhanced with parallel PDF processing and batch embedding.
"""

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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
    Enhanced with parallel PDF processing and batch embedding.
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
            # Check if collection exists
            if not utility.has_collection(self.collection_name):
                logger.info("Collection %s does not exist yet", self.collection_name)
                return

            # Get collection
            collection = Collection(self.collection_name)

            # Check if collection has data
            if collection.num_entities == 0:
                logger.info("Collection %s is empty", self.collection_name)
                return

            # Load collection to memory
            collection.load()

            # Wait a bit for collection to be ready
            import time

            time.sleep(0.5)

            # Query all unique paper IDs using batch querying for large collections
            batch_size = 1000
            all_paper_ids = set()
            offset = 0

            logger.info(
                "Starting to sync paper IDs from collection %s", self.collection_name
            )

            while True:
                try:
                    # Query with pagination
                    # Use a more specific expression to ensure paper_id field exists
                    results = collection.query(
                        expr="paper_id >= ''",  # This ensures paper_id exists and is not null
                        output_fields=["paper_id"],
                        offset=offset,
                        limit=batch_size,
                    )

                    if not results:
                        break

                    # Extract paper IDs from this batch
                    for result in results:
                        paper_id = result.get("paper_id")
                        if paper_id:
                            all_paper_ids.add(paper_id)

                    # If we got fewer results than batch_size, we've reached the end
                    if len(results) < batch_size:
                        break

                    offset += batch_size

                    # Log progress for large collections
                    if offset % 5000 == 0:
                        logger.info(
                            "Processed %d entities, found %d unique papers so far",
                            offset,
                            len(all_paper_ids),
                        )

                except Exception as batch_error:
                    logger.warning(
                        "Error in batch %d: %s", offset // batch_size, batch_error
                    )
                    break

            self.loaded_papers = all_paper_ids
            logger.info(
                "Successfully synced %d loaded papers from collection %s",
                len(self.loaded_papers),
                self.collection_name,
            )

            # Log first few paper IDs for debugging
            if self.loaded_papers:
                sample_ids = list(self.loaded_papers)[:5]
                logger.debug("Sample paper IDs: %s", sample_ids)

        except Exception as e:
            logger.error("Failed to sync loaded papers: %s", e)
            logger.info("Starting with empty loaded_papers set")

    def has_paper(self, paper_id: str) -> bool:
        """
        Check if a paper is already loaded in the vector store.

        Args:
            paper_id: The ID of the paper to check

        Returns:
            bool: True if paper exists in vector store, False otherwise
        """
        return paper_id in self.loaded_papers

    def _load_and_split_pdf(
        self, paper_id: str, pdf_url: str, paper_metadata: Dict[str, Any]
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
        if self.config is None:
            raise ValueError(
                "Configuration is required for text splitting in Vectorstore."
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
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
            for field in self.metadata_fields:
                if field in paper_metadata and field not in chunk.metadata:
                    chunk.metadata[field] = paper_metadata[field]

            # Store in local dict for compatibility
            self.documents[chunk_id] = chunk

        return chunks

    def add_papers_batch(
        self,
        papers_to_add: List[Tuple[str, str, Dict[str, Any]]],
        max_workers: int = 5,
        batch_size: int = 100,
    ) -> None:
        """
        Add multiple papers to the document store in parallel with batch embedding.

        Args:
            papers_to_add: List of tuples (paper_id, pdf_url, paper_metadata)
            max_workers: Maximum number of parallel PDF loading workers
            batch_size: Number of chunks to embed in a single batch
        """
        if not papers_to_add:
            logger.info("No papers to add")
            return

        # Filter out already loaded papers BEFORE processing
        papers_to_process = []
        for paper_id, pdf_url, metadata in papers_to_add:
            if paper_id in self.loaded_papers:
                logger.debug("Paper %s already loaded, skipping", paper_id)
            else:
                papers_to_process.append((paper_id, pdf_url, metadata))

        if not papers_to_process:
            logger.info("All %d papers are already loaded", len(papers_to_add))
            return

        logger.info(
            "Starting PARALLEL batch processing of %d papers with %d workers",
            len(papers_to_process),
            max_workers,
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
                    self._load_and_split_pdf, paper_id, pdf_url, metadata
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
                    self.paper_metadata[paper_id] = metadata

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
                "Starting BATCH EMBEDDING of %d chunks in batches of %d",
                total_chunks,
                batch_size,
            )

            # Process in batches
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                batch_chunks = all_chunks[i:batch_end]
                batch_ids = all_ids[i:batch_end]

                logger.info(
                    "Processing embedding batch %d/%d (chunks %d-%d of %d)",
                    (i // batch_size) + 1,
                    (total_chunks + batch_size - 1) // batch_size,
                    i + 1,
                    batch_end,
                    total_chunks,
                )

                # Extract texts for embedding
                texts = [chunk.page_content for chunk in batch_chunks]

                # Log embedding API call
                logger.info(
                    "Calling embedding API for batch of %d texts (avg length: %d chars)",
                    len(texts),
                    sum(len(t) for t in texts) // len(texts) if texts else 0,
                )

                # Add to Milvus (this will trigger embedding)
                self.vector_store.add_documents(
                    documents=batch_chunks,
                    ids=batch_ids,
                )

                logger.info(
                    "Successfully embedded and stored batch %d/%d",
                    (i // batch_size) + 1,
                    (total_chunks + batch_size - 1) // batch_size,
                )

            embed_time = time.time() - embed_start
            logger.info(
                "BATCH EMBEDDING COMPLETE: Embedded %d chunks in %.2f seconds (%.2f chunks/sec)",
                total_chunks,
                embed_time,
                total_chunks / embed_time if embed_time > 0 else 0,
            )

            # Update loaded papers
            for paper_id in successful_papers:
                self.loaded_papers.add(paper_id)

            total_time = time.time() - start_time
            logger.info(
                "FULL BATCH PROCESSING COMPLETE: %d papers, %d chunks in %.2f seconds total (%.2f sec/paper)",
                len(successful_papers),
                total_chunks,
                total_time,
                total_time / len(successful_papers) if successful_papers else 0,
            )

        except Exception as e:
            logger.error("Failed to add chunks to Milvus: %s", e)
            raise

    def add_paper(
        self,
        paper_id: str,
        pdf_url: str,
        paper_metadata: Dict[str, Any],
    ) -> None:
        """
        Add a single paper to the document store.
        For backward compatibility - internally uses batch method.

        Args:
            paper_id: Unique identifier for the paper
            pdf_url: URL to the PDF
            paper_metadata: Metadata about the paper
        """
        # Skip if already loaded
        if paper_id in self.loaded_papers:
            logger.info("Paper %s already loaded (found in memory), skipping", paper_id)
            return

        # Use batch method for single paper
        self.add_papers_batch([(paper_id, pdf_url, paper_metadata)], max_workers=1)

    def _is_paper_in_milvus(self, paper_id: str) -> bool:
        """Check if a paper exists in Milvus by querying for its chunks."""
        try:
            if not utility.has_collection(self.collection_name):
                return False

            # Query for any chunk from this paper
            results = self.vector_store.similarity_search(
                query="", k=1, expr=f'paper_id == "{paper_id}"'  # Empty query
            )

            return len(results) > 0

        except Exception as e:
            logger.debug("Error checking paper %s in Milvus: %s", paper_id, e)
            return False

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

                # Get unique paper count by sampling
                sample_papers = set()
                if collection.num_entities > 0:
                    try:
                        results = collection.query(
                            expr="paper_id >= ''",
                            output_fields=["paper_id"],
                            limit=min(1000, collection.num_entities),
                        )
                        for result in results:
                            if "paper_id" in result:
                                sample_papers.add(result["paper_id"])
                    except:
                        pass

                stats = {
                    "name": self.collection_name,
                    "num_entities": collection.num_entities,
                    "loaded_papers_in_memory": list(self.loaded_papers)[
                        :10
                    ],  # First 10 for brevity
                    "num_loaded_papers_in_memory": len(self.loaded_papers),
                    "sample_papers_in_milvus": list(sample_papers)[:10],
                    "estimated_papers_in_milvus": len(sample_papers),
                }
                return stats
            else:
                return {"error": f"Collection {self.collection_name} does not exist"}
        except MilvusException as e:
            logger.error("Failed to get collection stats: %s", e)
            return {"error": str(e)}
