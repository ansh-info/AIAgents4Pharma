"""
Shared Vector Store Manager for managing a single instance across Streamlit and LangGraph tools.
"""

import logging
from typing import Optional, Any
from threading import Lock

from langchain_core.embeddings import Embeddings
from .vector_store import Vectorstore

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Singleton manager for the Vectorstore instance.
    Ensures only one instance is used across the entire application.
    """

    _instance: Optional["VectorStoreManager"] = None
    _lock = Lock()
    _vector_store: Optional[Vectorstore] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self, embedding_model: Embeddings, config: Any, force_reinit: bool = False
    ) -> Vectorstore:
        """
        Initialize or get the vector store instance.

        Args:
            embedding_model: The embedding model to use
            config: Configuration object
            force_reinit: Force re-initialization even if instance exists

        Returns:
            The Vectorstore instance
        """
        with self._lock:
            if self._vector_store is None or force_reinit:
                logger.info("Initializing new Vectorstore instance")
                self._vector_store = Vectorstore(
                    embedding_model=embedding_model, config=config
                )
            else:
                logger.info("Using existing Vectorstore instance")

            return self._vector_store

    def get_instance(self) -> Optional[Vectorstore]:
        """
        Get the current vector store instance if it exists.

        Returns:
            The Vectorstore instance or None if not initialized
        """
        return self._vector_store

    def reset(self):
        """Reset the vector store instance."""
        with self._lock:
            self._vector_store = None
            logger.info("Vector store instance reset")


# Global manager instance
vector_store_manager = VectorStoreManager()
