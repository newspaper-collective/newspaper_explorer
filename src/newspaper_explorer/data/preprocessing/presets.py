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

from typing import Any, Union

# Step type: either a string (step name) or dict with name and args
# String example: "normalize_unicode"
# Dict example: {"name": "normalize_whitespace", "args": {"keep_newlines": True}}
StepType = Union[str, dict[str, Any]]

# General-purpose preprocessing pipelines
GENERAL_PIPELINES: dict[str, dict[str, Union[str, list[StepType]]]] = {
    "minimal": {
        "description": "Minimal processing, preserves original text as much as possible",
        "steps": [
            "normalize_unicode",
            {"name": "normalize_whitespace", "args": {"keep_newlines": True}},
        ],
        "use_case": "When you need to preserve original spelling and formatting",
    },
    "basic": {
        "description": "Light OCR cleanup, preserves original word forms",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "filter_empty_lines",
        ],
        "use_case": "Quick cleanup for raw OCR data while keeping original spelling",
    },
    "standard": {
        "description": "Standard preprocessing for most analysis tasks",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
        ],
        "use_case": "General text analysis, NER, embeddings, topic modeling",
    },
    "advanced": {
        "description": "Aggressive cleanup with lowercase and OCR quality filtering",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            "calculate_quality_metrics",
            {"name": "filter_by_quality_score", "args": {"min_quality": "review"}},
            "normalize_casing",
        ],
        "use_case": "When OCR quality is poor and aggressive filtering is needed",
    },
    "full": {
        "description": "Maximum cleaning for bag-of-words analysis",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            "calculate_quality_metrics",
            {"name": "filter_by_quality_score", "args": {"min_quality": "review"}},
            "normalize_casing",
            "remove_punctuation",
            "remove_numbers",
            "remove_stopwords",
        ],
        "use_case": "Word frequency analysis, keyword extraction, TF-IDF",
    },
}


# Analysis-specific preprocessing pipelines
# These are task-optimized variants that may skip or add specific steps
ANALYSIS_PIPELINES: dict[str, dict[str, Union[str, list[StepType]]]] = {
    "entities": {
        "description": "Optimized for named entity recognition and extraction",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "filter_lines_without_alphabetic_chars",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            {"name": "filter_by_total_character_length", "args": {"min_length": 10}},
            # Note: NO lowercase - entities are case-sensitive
            # Note: NO punctuation removal - needed for abbreviations (Dr., Inc.)
        ],
        "use_case": "Named entity recognition (NER), person/org/location extraction",
    },
    "topics": {
        "description": "Optimized for topic modeling (BERTopic, etc.)",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "filter_lines_without_alphabetic_chars",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            {"name": "filter_by_total_character_length", "args": {"min_length": 10}},
            "normalize_casing",
            "remove_stopwords",
            "remove_short_words",
            "normalize_whitespace",
        ],
        "use_case": "Topic modeling, document clustering, thematic analysis",
    },
    "emotions": {
        "description": "Optimized for emotion classification and sentiment analysis",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "filter_lines_without_alphabetic_chars",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            {"name": "filter_by_total_character_length", "args": {"min_length": 10}},
            # Note: NO lowercase - preserves emphasis (ANGRY vs angry)
            # Note: Keep punctuation - ! and ? matter for emotions
            # Note: Keep stopwords - "nicht gut" vs "gut" has different sentiment
        ],
        "use_case": "Emotion classification, sentiment analysis, affect detection",
    },
    "keywords": {
        "description": "Optimized for keyword extraction and term frequency analysis",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "filter_lines_without_alphabetic_chars",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            {"name": "filter_by_total_character_length", "args": {"min_length": 10}},
            "normalize_casing",
            "remove_punctuation",
            "remove_numbers",
            "remove_stopwords",
            "remove_short_words",
            "normalize_whitespace",
        ],
        "use_case": "Keyword extraction, TF-IDF, important term identification",
    },
    "embeddings": {
        "description": "Optimized for generating text embeddings",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "filter_lines_without_alphabetic_chars",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            {"name": "filter_by_total_character_length", "args": {"min_length": 10}},
            # Note: Keep case, punctuation - models are trained on natural text
        ],
        "use_case": "Sentence/document embeddings, semantic similarity, vector search",
    },
    "concepts": {
        "description": "Optimized for concept extraction and semantic analysis",
        "steps": [
            "normalize_unicode",
            "normalize_long_s",
            "normalize_whitespace",
            "only_keep_allowed_chars",
            "dehyphenate",
            "filter_empty_lines",
            "filter_lines_without_alphabetic_chars",
            "remove_garbage_words",
            {"name": "filter_by_word_count", "args": {"min_words": 2}},
            {"name": "filter_by_total_character_length", "args": {"min_length": 10}},
            "normalize_casing",
        ],
        "use_case": "Concept extraction, semantic network analysis, knowledge graphs",
    },
}


# All available pipelines
ALL_PIPELINES: dict[str, dict[str, Union[str, list[StepType]]]] = {
    **GENERAL_PIPELINES,
    **ANALYSIS_PIPELINES,
}


def get_preset(name: str) -> list[StepType]:
    """
    Get a preprocessing pipeline preset by name.

    Returns a list of steps that can be passed directly to TextPreprocessor.pipeline().
    Steps can be either strings (step name) or dicts with 'name' and 'args' keys.

    Available presets:

    General-purpose (progressive complexity):
    - minimal: Preserve original text, only fix encoding and whitespace
    - basic: Light OCR cleanup (+ long_s normalization)
    - standard: General analysis ready (+ dehyphenate, remove invalid chars)
    - advanced: Aggressive cleanup (+ lowercase, OCR quality filters)
    - full: Maximum cleaning for bag-of-words (+ remove punct/numbers/stopwords)

    Analysis-specific (task-optimized):
    - entities: NER (no lowercase, keeps punctuation for abbreviations)
    - topics: Topic modeling (lowercase, removes stopwords)
    - emotions: Sentiment analysis (no lowercase, keeps punct for ! and ?)
    - keywords: TF-IDF (removes punct, numbers, stopwords)
    - embeddings: Vector embeddings (keeps case and punctuation)
    - concepts: Concept extraction (lowercase)

    Args:
        name: Pipeline preset name

    Returns:
        List of preprocessing steps (strings or dicts with args)

    Raises:
        ValueError: If preset name is unknown

    Example:
        >>> from newspaper_explorer.data.preprocessing.presets import get_preset
        >>> from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
        >>>
        >>> preprocessor = TextPreprocessor(text_column="text")
        >>> steps = get_preset("standard")
        >>> df_processed = preprocessor.pipeline(df, steps=steps)
        >>>
        >>> # Steps with args are also supported:
        >>> steps = get_preset("minimal")  # includes {"name": "normalize_whitespace", "args": {"keep_newlines": True}}
    """
    if name not in ALL_PIPELINES:
        available = ", ".join(sorted(ALL_PIPELINES.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    steps = ALL_PIPELINES[name]["steps"]
    # Ensure we always return a list
    if isinstance(steps, str):  # pragma: no cover
        return [steps]
    return list(steps)


def list_presets(category: str = "all") -> dict[str, dict[str, Union[str, list[StepType]]]]:
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
    if category == "general":
        return GENERAL_PIPELINES.copy()
    if category == "analysis":
        return ANALYSIS_PIPELINES.copy()
    raise ValueError(f"Unknown category '{category}'. Use 'all', 'general', or 'analysis'")


def get_preset_info(name: str) -> dict[str, Union[str, list[StepType]]]:
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
        >>> print(f"Use case: {info['use_case']}")
        >>> print(f"Steps: {len(info['steps'])}")
    """
    if name not in ALL_PIPELINES:
        available = ", ".join(sorted(ALL_PIPELINES.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return ALL_PIPELINES[name].copy()


def get_extra_steps(preset_name: str, base_preset: str = "standard") -> list[StepType]:
    """
    Get steps in a preset that are not already in the base preset.

    Useful for chaining: run the base preset first, then only the extra
    steps from a method-specific preset on top of the base output.

    Args:
        preset_name: The target preset to get extra steps from
        base_preset: The base preset to subtract (default: "standard")

    Returns:
        List of steps in preset_name that are not in base_preset.
        Returns empty list if preset is identical to base.

    Example:
        >>> get_extra_steps("keywords")
        ['normalize_casing', 'remove_punctuation', 'remove_numbers', 'remove_stopwords']
        >>> get_extra_steps("entities")
        []
    """
    base_steps = get_preset(base_preset)
    preset_steps = get_preset(preset_name)
    base_step_names = {get_step_name(s) for s in base_steps}
    return [s for s in preset_steps if get_step_name(s) not in base_step_names]


def get_step_name(step: StepType) -> str:
    """
    Extract the step name from a step (string or dict).

    Args:
        step: Either a string step name or dict with 'name' key

    Returns:
        The step name as a string

    Example:
        >>> get_step_name("normalize_unicode")
        'normalize_unicode'
        >>> get_step_name({"name": "normalize_whitespace", "args": {"keep_newlines": True}})
        'normalize_whitespace'
    """
    if isinstance(step, str):
        return step
    return str(step["name"])


def get_step_args(step: StepType) -> dict[str, Any]:
    """
    Extract the arguments from a step.

    Args:
        step: Either a string step name or dict with 'name' and optional 'args' keys

    Returns:
        Dictionary of arguments (empty dict if step is a string or has no args)

    Example:
        >>> get_step_args("normalize_unicode")
        {}
        >>> get_step_args({"name": "normalize_whitespace", "args": {"keep_newlines": True}})
        {'keep_newlines': True}
    """
    if isinstance(step, str):
        return {}
    args = step.get("args", {})
    return dict(args) if args else {}
