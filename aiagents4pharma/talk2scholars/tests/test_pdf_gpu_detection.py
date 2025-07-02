import pytest
from unittest.mock import patch, MagicMock

from aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection import (
    detect_nvidia_gpu,
    get_optimal_index_config,
    log_index_configuration,
)


# === detect_nvidia_gpu ===


def test_detect_nvidia_gpu_force_cpu_from_config():
    class GPUConfig:
        force_cpu_mode = True

    class Config:
        gpu_detection = GPUConfig()

    assert detect_nvidia_gpu(Config()) is False


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.subprocess.run")
def test_detect_nvidia_gpu_success(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0, stdout="NVIDIA A100\nNVIDIA RTX 3090"
    )

    assert detect_nvidia_gpu() is True
    mock_run.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.subprocess.run")
def test_detect_nvidia_gpu_no_output(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="")

    assert detect_nvidia_gpu() is False


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.subprocess.run")
def test_detect_nvidia_gpu_exception(mock_run):
    mock_run.side_effect = RuntimeError("command failed")
    assert detect_nvidia_gpu() is False


# === get_optimal_index_config ===


def test_get_optimal_index_config_gpu():
    index_params, search_params = get_optimal_index_config(
        has_gpu=True, embedding_dim=768
    )

    assert index_params["index_type"] == "GPU_CAGRA"
    assert "cache_dataset_on_device" in index_params["params"]
    assert search_params["params"]["search_width"] == 16


def test_get_optimal_index_config_cpu():
    index_params, search_params = get_optimal_index_config(
        has_gpu=False, embedding_dim=768
    )

    assert index_params["index_type"] == "IVF_FLAT"
    assert index_params["params"]["nlist"] == 96  # 768 / 8 = 96
    assert search_params["params"]["nprobe"] == 16


# === log_index_configuration ===


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.logger")
def test_log_index_configuration_logs_all(mock_logger):
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    log_index_configuration(index_params, search_params)

    assert mock_logger.info.call_count >= 5
