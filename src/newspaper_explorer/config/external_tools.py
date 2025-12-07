"""
External tool cache directory configuration.

Sets environment variables for ML libraries BEFORE they are imported.
Import this module early in any file that uses HuggingFace, YOLO, etc.

Usage:
    from newspaper_explorer.config import external_tools  # noqa: F401
    from transformers import ...  # Now uses correct cache directory

Note:
    Uses os.environ.setdefault() so user-provided values in .env take precedence.

    spaCy models are NOT configurable via environment variables - they are
    installed as pip packages in site-packages. Use `python -m spacy download <model>`.
"""

import os

from newspaper_explorer.config.base import get_project_root

_project_root = get_project_root()
_cache_dir = _project_root / ".cache"

# HuggingFace / Transformers cache
os.environ.setdefault("HF_HOME", str(_cache_dir / "huggingface"))

# Sentence Transformers (KeyBERT, BERTopic, FASTopic)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_cache_dir / "sentence_transformers"))

# YOLO / Ultralytics
os.environ.setdefault("YOLO_CONFIG_DIR", str(_cache_dir / "ultralytics"))
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "False")

# Suppress noisy HuggingFace Hub warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
