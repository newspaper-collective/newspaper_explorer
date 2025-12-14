"""
Machine learning utilities.

General ML/PyTorch utilities for hardware detection and setup.
"""

from typing import Any

import torch


def check_cuda_available() -> dict[str, Any]:
    """
    Check CUDA/GPU availability and properties.

    Returns:
        Dictionary with cuda_available, device_count, and gpu_info list
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    gpu_info: list[dict[str, Any]] = []
    if cuda_available:
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(
                {
                    "id": i,
                    "name": props.name,  # type: ignore[attr-defined]
                    "total_memory_gb": props.total_memory / (1024**3),  # type: ignore[attr-defined]
                }
            )

    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpu_info": gpu_info,
    }
