"""
Milvus Connection Manager for Talk2KnowledgeGraphs.

This module provides centralized connection management for Milvus database,
removing the dependency on frontend session state and enabling proper
separation of concerns between frontend and backend.
"""

import logging
import threading
from typing import Optional, Dict, Any

import hydra
from pymilvus import connections, db, Collection
from pymilvus.exceptions import ConnectionNotExistException, MilvusException

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusConnectionManager:
    """
    Centralized Milvus connection manager for backend tools with singleton pattern.

    This class handles:
    - Connection establishment and management
    - Database switching
    - Connection health checks
    - Graceful error handling
    - Thread-safe singleton pattern

    Args:
        cfg: Configuration object containing Milvus connection parameters
    """

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, cfg: Dict[str, Any]):
        """
        Create singleton instance based on database configuration.

        Args:
            cfg: Configuration dictionary containing Milvus DB settings

        Returns:
            MilvusConnectionManager: Singleton instance for the given config
        """
        # Create a unique key based on connection parameters
        config_key = (
            cfg.milvus_db.host,
            int(cfg.milvus_db.port),
            cfg.milvus_db.user,
            cfg.milvus_db.database_name,
            cfg.milvus_db.alias,
        )

        if config_key not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if config_key not in cls._instances:
                    instance = super(MilvusConnectionManager, cls).__new__(cls)
                    cls._instances[config_key] = instance
                    logger.info(
                        "Created new MilvusConnectionManager singleton for database: %s",
                        cfg.milvus_db.database_name,
                    )
        else:
            logger.debug(
                "Reusing existing MilvusConnectionManager singleton for database: %s",
                cfg.milvus_db.database_name,
            )

        return cls._instances[config_key]

    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the Milvus connection manager.

        Args:
            cfg: Configuration dictionary containing Milvus DB settings
        """
        # Prevent re-initialization of singleton instance
        if hasattr(self, "_initialized"):
            return

        self.cfg = cfg
        self.alias = cfg.milvus_db.alias
        self.host = cfg.milvus_db.host
        self.port = int(cfg.milvus_db.port)  # Ensure port is integer
        self.user = cfg.milvus_db.user
        self.password = cfg.milvus_db.password
        self.database_name = cfg.milvus_db.database_name

        # Thread lock for connection operations
        self._connection_lock = threading.Lock()

        # Mark as initialized
        self._initialized = True

        logger.info(
            "MilvusConnectionManager initialized for database: %s", self.database_name
        )

    def ensure_connection(self) -> bool:
        """
        Ensure Milvus connection exists, create if not.

        This method checks if a connection with the specified alias exists,
        and creates one if it doesn't. It also switches to the correct database.
        Thread-safe implementation with connection locking.

        Returns:
            bool: True if connection is established, False otherwise

        Raises:
            MilvusException: If connection cannot be established
        """
        with self._connection_lock:
            try:
                # Check if connection already exists
                if not connections.has_connection(self.alias):
                    logger.info(
                        "Creating new Milvus connection with alias: %s", self.alias
                    )
                    connections.connect(
                        alias=self.alias,
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                    )
                    logger.info(
                        "Successfully connected to Milvus at %s:%s",
                        self.host,
                        self.port,
                    )
                else:
                    logger.debug(
                        "Milvus connection already exists with alias: %s", self.alias
                    )

                # Switch to the correct database
                db.using_database(self.database_name)
                logger.debug("Using Milvus database: %s", self.database_name)

                return True

            except MilvusException as e:
                logger.error("Failed to establish Milvus connection: %s", str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error during Milvus connection: %s", str(e))
                raise MilvusException(f"Connection failed: {str(e)}")

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get current connection information.

        Returns:
            Dict containing connection details
        """
        try:
            if connections.has_connection(self.alias):
                conn_addr = connections.get_connection_addr(self.alias)
                return {
                    "alias": self.alias,
                    "host": self.host,
                    "port": self.port,
                    "database": self.database_name,
                    "connected": True,
                    "connection_address": conn_addr,
                }
            else:
                return {
                    "alias": self.alias,
                    "host": self.host,
                    "port": self.port,
                    "database": self.database_name,
                    "connected": False,
                    "connection_address": None,
                }
        except Exception as e:
            logger.error("Error getting connection info: %s", str(e))
            return {"alias": self.alias, "connected": False, "error": str(e)}

    def test_connection(self) -> bool:
        """
        Test the connection by attempting to list collections.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            self.ensure_connection()

            # Try to get a collection to test the connection
            test_collection_name = f"{self.database_name}_nodes"
            Collection(name=test_collection_name)

            logger.debug("Connection test successful")
            return True

        except Exception as e:
            logger.error("Connection test failed: %s", str(e))
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from Milvus.

        Returns:
            bool: True if disconnected successfully, False otherwise
        """
        try:
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
                logger.info("Disconnected from Milvus with alias: %s", self.alias)
                return True
            else:
                logger.debug("No connection to disconnect with alias: %s", self.alias)
                return True

        except Exception as e:
            logger.error("Error disconnecting from Milvus: %s", str(e))
            return False

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a Milvus collection, ensuring connection is established.
        Thread-safe implementation.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            Collection: The requested Milvus collection

        Raises:
            MilvusException: If collection cannot be retrieved
        """
        try:
            self.ensure_connection()
            collection = Collection(name=collection_name)
            collection.load()  # Load collection data
            logger.debug("Successfully loaded collection: %s", collection_name)
            return collection

        except Exception as e:
            logger.error("Failed to get collection %s: %s", collection_name, str(e))
            raise MilvusException(
                f"Failed to get collection {collection_name}: {str(e)}"
            )

    @classmethod
    def get_instance(cls, cfg: Dict[str, Any]) -> "MilvusConnectionManager":
        """
        Get singleton instance for the given configuration.

        Args:
            cfg: Configuration dictionary containing Milvus DB settings

        Returns:
            MilvusConnectionManager: Singleton instance for the given config
        """
        return cls(cfg)

    @classmethod
    def clear_instances(cls):
        """
        Clear all singleton instances. Useful for testing or cleanup.
        """
        with cls._lock:
            # Disconnect all existing connections before clearing
            for instance in cls._instances.values():
                instance.disconnect()
            cls._instances.clear()
            logger.info("Cleared all MilvusConnectionManager singleton instances")

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MilvusConnectionManager":
        """
        Create a MilvusConnectionManager from configuration.

        Args:
            cfg: Configuration object or dictionary

        Returns:
            MilvusConnectionManager: Configured connection manager instance
        """
        return cls(cfg)

    @classmethod
    def from_hydra_config(
        cls,
        config_path: str = "../configs",
        config_name: str = "config",
        overrides: Optional[list] = None,
    ) -> "MilvusConnectionManager":
        """
        Create a MilvusConnectionManager from Hydra configuration.

        This method loads the Milvus database configuration using Hydra,
        providing complete backend separation from frontend configs.

        Args:
            config_path: Path to the configs directory
            config_name: Name of the main config file
            overrides: List of config overrides

        Returns:
            MilvusConnectionManager: Configured connection manager instance

        Example:
            # Load with default database config
            conn_manager = MilvusConnectionManager.from_hydra_config()

            # Load with specific overrides
            conn_manager = MilvusConnectionManager.from_hydra_config(
                overrides=["utils/database/milvus=default"]
            )
        """
        if overrides is None:
            overrides = ["utils/database/milvus=default"]

        try:
            with hydra.initialize(version_base=None, config_path=config_path):
                cfg_all = hydra.compose(config_name=config_name, overrides=overrides)
                cfg = (
                    cfg_all.utils.database.milvus
                )  # Extract utils.database.milvus section
                logger.info(
                    "Loaded Milvus config from Hydra with overrides: %s", overrides
                )
                return cls(cfg)
        except Exception as e:
            logger.error("Failed to load Hydra configuration: %s", str(e))
            raise MilvusException(f"Configuration loading failed: {str(e)}")
