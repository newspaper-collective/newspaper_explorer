"""
Preprocessing pipeline presets.

This module defines recommended preprocessing pipelines for different use cases.
Each preset is a configuration dictionary with steps, description, and use case.

Presets can be used via CLI:
    newspaper-explorer data preprocess --source der_tag --pipeline standard

Or programmatically:
    from newspaper_explorer.data.preprocessing.presets import get_preset
    steps = get_preset("standard")
"""

from typing import Union

# General-purpose preprocessing pipelines
GENERAL_PIPELINES: dict[str, dict[str, Union[str, list[str]]]] = {
    "minimal": {
        "description": "Minimal processing, preserves original text as much as possible",
        "steps": [
            "normalize-unicode",  # Only fix critical OCR issues
            "clean-ocr",  # Remove invalid characters
        ],
        "use_case": "When you need to preserve original spelling and formatting",
    },
    "basic": {
        "description": "Basic OCR cleanup and normalization without heavy processing",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
        ],
        "use_case": "Fast cleanup for raw OCR data, preserves original word forms",
    },
    "standard": {
        "description": "Standard preprocessing for most analysis tasks",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters (ſ→s, etc.)
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            "lowercase",  # Convert to lowercase
        ],
        "use_case": "General text analysis, topic modeling, embeddings",
    },
    "search": {
        "description": "Optimized for search and matching",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "remove-diacritics",  # ä→a, ö→o, etc. for better matching
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            "lowercase",  # Convert to lowercase
        ],
        "use_case": "Full-text search, fuzzy matching, entity extraction",
    },
    "analysis": {
        "description": "Prepared for NLP analysis with filtering",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            "lowercase",  # Convert to lowercase
            "remove-punctuation",  # Remove punctuation
            "remove-numbers",  # Remove numbers
            "remove-stopwords",  # Remove German stopwords
        ],
        "use_case": "Word frequency analysis, keyword extraction, statistical analysis",
    },
}


# Analysis-specific preprocessing pipelines
ANALYSIS_PIPELINES: dict[str, dict[str, Union[str, list[str]]]] = {
    "entities": {
        "description": "Optimized for named entity recognition and extraction",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens (preserves entity boundaries)
            # Note: NO lowercase - entities are case-sensitive
            # Note: NO punctuation removal - needed for abbreviations (Dr., Inc.)
        ],
        "use_case": "Named entity recognition (NER), person/org/location extraction",
    },
    "topics": {
        "description": "Optimized for topic modeling (BERTopic, etc.)",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            "lowercase",  # Convert to lowercase
            "remove-stopwords",  # Remove stopwords (improves topic coherence)
        ],
        "use_case": "Topic modeling, document clustering, thematic analysis",
    },
    "emotions": {
        "description": "Optimized for emotion classification and sentiment analysis",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            "lowercase",  # Convert to lowercase
            # Note: Keep punctuation - ! and ? matter for emotions
            # Note: Keep stopwords - "nicht gut" vs "gut" has different sentiment
        ],
        "use_case": "Emotion classification, sentiment analysis, affect detection",
    },
    "keywords": {
        "description": "Optimized for keyword extraction and term frequency analysis",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            "lowercase",  # Convert to lowercase
            "remove-punctuation",  # Remove punctuation
            "remove-numbers",  # Remove numbers
            "remove-stopwords",  # Remove stopwords
        ],
        "use_case": "Keyword extraction, TF-IDF, important term identification",
    },
    "embeddings": {
        "description": "Optimized for generating text embeddings",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            # Note: Keep case, punctuation - models are trained on natural text
        ],
        "use_case": "Sentence/document embeddings, semantic similarity, vector search",
    },
    "concepts": {
        "description": "Optimized for concept extraction and semantic analysis",
        "steps": [
            "normalize-unicode",  # Fix OCR artifacts, unify quotes/hyphens
            "normalize",  # Normalize historical German characters
            "normalize-whitespace",  # Clean up whitespace
            "clean-ocr",  # Remove OCR artifacts
            "dehyphenate",  # Remove line-break hyphens
            "lowercase",  # Convert to lowercase
        ],
        "use_case": "Concept extraction, semantic network analysis, knowledge graphs",
    },
}


# All available pipelines
ALL_PIPELINES: dict[str, dict[str, Union[str, list[str]]]] = {
    **GENERAL_PIPELINES,
    **ANALYSIS_PIPELINES,
}


def get_preset(name: str) -> list[str]:
    """
    Get a preprocessing pipeline preset by name.

    Available presets:

    General-purpose:
    - minimal: Preserve original text, only fix critical OCR issues
    - basic: Fast OCR cleanup (recommended start)
    - standard: General text analysis (default choice)
    - search: Optimized for search and matching
    - analysis: Word frequency with filtering

    Analysis-specific:
    - entities: Named entity recognition (NER)
    - topics: Topic modeling (BERTopic, etc.)
    - emotions: Emotion classification, sentiment analysis
    - keywords: Keyword extraction, TF-IDF
    - embeddings: Text embeddings, semantic similarity
    - concepts: Concept extraction, semantic networks

    Args:
        name: Pipeline preset name

    Returns:
        List of preprocessing step names

    Raises:
        ValueError: If preset name is unknown

    Example:
        >>> from newspaper_explorer.data.preprocessing.presets import get_preset
        >>> from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
        >>>
        >>> preprocessor = TextPreprocessor(text_column="text")
        >>> steps = get_preset("entities")
        >>> df_processed = preprocessor.pipeline(df, steps=steps)
    """
    if name not in ALL_PIPELINES:
        available = ", ".join(sorted(ALL_PIPELINES.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return ALL_PIPELINES[name]["steps"]


def list_presets(category: str = "all") -> dict[str, dict[str, Union[str, list[str]]]]:
    """
    List available preprocessing pipeline presets.

    Args:
        category: Which presets to list ("all", "general", or "analysis")

    Returns:
        Dictionary mapping preset names to their configuration

    Example:
        >>> from newspaper_explorer.data.preprocessing.presets import list_presets
        >>>
        >>> # List all presets
        >>> presets = list_presets()
        >>>
        >>> # List only analysis-specific presets
        >>> analysis_presets = list_presets(category="analysis")
        >>>
        >>> for name, config in analysis_presets.items():
        ...     print(f"{name}: {config['description']}")
    """
    if category == "all":
        return ALL_PIPELINES.copy()
    elif category == "general":
        return GENERAL_PIPELINES.copy()
    elif category == "analysis":
        return ANALYSIS_PIPELINES.copy()
    else:
        raise ValueError(f"Unknown category '{category}'. Use 'all', 'general', or 'analysis'")


def get_preset_info(name: str) -> dict[str, Union[str, list[str]]]:
    """
    Get detailed information about a specific preset.

    Args:
        name: Pipeline preset name

    Returns:
        Dictionary with preset configuration (description, steps, use_case)

    Example:
        >>> from newspaper_explorer.data.preprocessing.presets import get_preset_info
        >>>
        >>> info = get_preset_info("entities")
        >>> print(info["description"])
        >>> print(f"Steps: {', '.join(info['steps'])}")
    """
    if name not in ALL_PIPELINES:
        available = ", ".join(sorted(ALL_PIPELINES.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return ALL_PIPELINES[name].copy()
