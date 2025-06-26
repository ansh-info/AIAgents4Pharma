"""
GPU Detection Utility for Milvus Index Selection
Detects NVIDIA GPU availability and determines appropriate index configuration
"""

import logging
import subprocess
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def detect_nvidia_gpu() -> bool:
    """
    Detect if NVIDIA GPU is available and accessible.

    Returns:
        bool: True if NVIDIA GPU is detected and accessible, False otherwise
    """
    try:
        # Check if nvidia-smi command is available
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_names = result.stdout.strip().split("\n")
            logger.info("Detected NVIDIA GPU(s): %s", gpu_names)
            return True
        else:
            logger.info("nvidia-smi command failed or no GPUs detected")
            return False

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ) as e:
        logger.info("NVIDIA GPU detection failed: %s", e)
        return False
    except Exception as e:
        logger.warning("Unexpected error during GPU detection: %s", e)
        return False


def get_optimal_index_config(
    has_gpu: bool, embedding_dim: int = 768
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get optimal index and search parameters based on GPU availability.

    Args:
        has_gpu (bool): Whether NVIDIA GPU is available
        embedding_dim (int): Dimension of embeddings

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: (index_params, search_params)
    """
    if has_gpu:
        logger.info("Configuring GPU_CAGRA index for NVIDIA GPU")

        # GPU_CAGRA index parameters - optimized for performance
        index_params = {
            "index_type": "GPU_CAGRA",
            "metric_type": "COSINE",  # Using COSINE as in original config
            "params": {
                "intermediate_graph_degree": 64,  # Higher for better recall
                "graph_degree": 32,  # Balanced performance/recall
                "build_algo": "IVF_PQ",  # Higher quality build
                "cache_dataset_on_device": "true",  # Cache for better recall
                "adapt_for_cpu": "false",  # Pure GPU mode
            },
        }

        # GPU_CAGRA search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "itopk_size": 128,  # Power of 2, good for intermediate results
                "search_width": 16,  # Balanced entry points
                "team_size": 16,  # Optimize for typical vector dimensions
            },
        }

    else:
        logger.info("Configuring CPU index (IVF_FLAT) - no NVIDIA GPU detected")

        # CPU IVF_FLAT index parameters (original configuration)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {
                "nlist": min(
                    1024, max(64, embedding_dim // 8)
                )  # Dynamic nlist based on dimension
            },
        }

        # CPU search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},  # Slightly higher than original for better recall
        }

    return index_params, search_params


def log_index_configuration(
    index_params: Dict[str, Any], search_params: Dict[str, Any]
) -> None:
    """Log the selected index configuration for debugging."""
    index_type = index_params.get("index_type", "Unknown")
    metric_type = index_params.get("metric_type", "Unknown")

    logger.info("=== Milvus Index Configuration ===")
    logger.info("Index Type: %s", index_type)
    logger.info("Metric Type: %s", metric_type)
    logger.info("Index Params: %s", index_params.get("params", {}))
    logger.info("Search Params: %s", search_params.get("params", {}))
    logger.info("===================================")
