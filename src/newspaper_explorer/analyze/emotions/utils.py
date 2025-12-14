"""
Emotion analysis utilities.

Helper functions for emotion model validation and status checking.
"""

from pathlib import Path
from typing import Any, Optional

from newspaper_explorer.data.utils.ml import check_cuda_available
from newspaper_explorer.models.analysis.emotions import EMOTIONS


def check_emotion_models(model_dir: Path) -> dict[str, Any]:
    """
    Check availability of emotion model files.

    Args:
        model_dir: Directory containing emotion model .pt files

    Returns:
        Dictionary with model_dir, exists, models (dict of emotion -> file info), all_found
    """
    model_dir = Path(model_dir)
    models: dict[str, dict[str, Any]] = {}

    for emotion in EMOTIONS:
        model_file = model_dir / f"{emotion}.pt"
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024**2)
            models[emotion] = {
                "found": True,
                "path": str(model_file),
                "size_mb": size_mb,
            }
        else:
            models[emotion] = {
                "found": False,
                "path": str(model_file),
                "size_mb": 0.0,
            }

    all_found = all(model_info.get("found", False) for model_info in models.values())

    return {
        "model_dir": str(model_dir.absolute()),
        "exists": model_dir.exists(),
        "models": models,
        "all_found": all_found,
    }


def get_model_status(model_dir: Optional[Path] = None) -> dict[str, Any]:
    """
    Get complete model environment status (CUDA + models).

    Args:
        model_dir: Directory containing emotion models (default: models/emotions)

    Returns:
        Combined dictionary with cuda_info and model_info
    """
    if model_dir is None:
        model_dir = Path("models/emotions")

    return {
        "cuda_info": check_cuda_available(),
        "model_info": check_emotion_models(model_dir),
    }
