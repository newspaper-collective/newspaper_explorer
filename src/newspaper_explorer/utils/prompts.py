"""
Centralized prompt template management for LLM interactions.

Provides reusable prompt templates with variable substitution for consistent
LLM behavior across different analysis tasks.
"""

from typing import Dict, List


class PromptTemplate:
    """
    Reusable prompt template with variable substitution.

    Example:
        ```python
        template = PromptTemplate(
            system="You are a historian.",
            user="Analyze this text: {text}"
        )

        prompt = template.format(text="Berlin 1920")
        ```
    """

    def __init__(self, system: str = "", user: str = ""):
        """
        Initialize prompt template.

        Args:
            system: System message template (model behavior instructions).
            user: User message template (task-specific prompt).
        """
        self.system = system
        self.user = user

    def format(self, **kwargs) -> Dict[str, str]:
        """
        Format template with provided variables.

        Args:
            **kwargs: Variables to substitute in templates.

        Returns:
            Dict with 'system' and 'user' keys containing formatted prompts.
        """
        return {
            "system": self.system.format(**kwargs) if self.system else "",
            "user": self.user.format(**kwargs),
        }


# ============================================================================
# Entity Extraction Prompts
# ============================================================================

ENTITY_EXTRACTION = PromptTemplate(
    system="""You are an expert at named entity recognition in historical German newspaper texts.
Extract entities accurately, preserving original spelling and historical names.
Consider historical context (e.g., place names may differ from modern names).""",
    user="""Extract named entities from the following German newspaper text.

Text:
{text}

Return a JSON object with these fields:
- persons: List of person names (e.g., "Kaiser Wilhelm II", "Otto von Bismarck")
- locations: List of places (e.g., "Berlin", "Deutsches Reich", "Wien")
- organizations: List of organizations (e.g., "Reichstag", "Sozialdemokratische Partei")

Use empty lists if no entities found. Preserve historical spelling.""",
)


# ============================================================================
# Topic Analysis Prompts
# ============================================================================

TOPIC_CLASSIFICATION = PromptTemplate(
    system="""You are a historian specializing in early 20th century German newspaper content.
Classify articles into historical topics considering the time period and cultural context.""",
    user="""Classify the following German newspaper text into relevant topics.

Text:
{text}

Available topics:
{topics}

Return a JSON object with:
- primary_topic: The main topic (choose one from the list)
- secondary_topics: Additional relevant topics (list, can be empty)
- confidence: Confidence score 0.0-1.0 for primary topic

If text is too short or unclear, use "unclear" as primary_topic with low confidence.""",
)


TOPIC_GENERATION = PromptTemplate(
    system="""You are a historian analyzing German newspaper content from the early 20th century.
Generate concise, descriptive topic labels that capture the essence of the text.""",
    user="""Generate topic labels for the following German newspaper text.

Text:
{text}

Return a JSON object with:
- topics: List of 1-5 topic labels (e.g., "Politik", "Wirtschaft", "Lokales", "Kultur")
- main_theme: One-sentence summary of the main theme

Labels should be:
- In German
- Historically appropriate (consider the era)
- Concise (1-2 words each)""",
)


# ============================================================================
# Emotion/Sentiment Analysis Prompts
# ============================================================================

EMOTION_ANALYSIS = PromptTemplate(
    system="""You are an expert in sentiment and emotion analysis of historical texts.
Consider the writing style and rhetorical conventions of early 20th century German newspapers.""",
    user="""Analyze the emotional tone and sentiment of this German newspaper text.

Text:
{text}

Return a JSON object with:
- sentiment: "positive", "negative", "neutral", or "mixed"
- emotions: List of detected emotions (e.g., "pride", "fear", "anger", "hope", "sadness")
- intensity: Emotional intensity 0.0-1.0
- tone: Descriptive tone (e.g., "patriotic", "critical", "celebratory", "somber")

Consider historical context and period-appropriate language.""",
)


# ============================================================================
# Concept Extraction Prompts
# ============================================================================

CONCEPT_EXTRACTION = PromptTemplate(
    system="""You are a historian extracting key concepts from German newspaper texts.
Identify important ideas, themes, and concepts relevant to the historical period.""",
    user="""Extract key concepts from the following German newspaper text.

Text:
{text}

Return a JSON object with:
- concepts: List of important concepts/themes (e.g., "Modernisierung", "Nationalismus", "Industrialisierung")
- relationships: List of concept relationships (e.g., "Industrialisierung leads to Urbanisierung")

Each relationship should be a dict with:
- source: Source concept
- target: Target concept
- type: Relationship type ("leads_to", "causes", "contradicts", "supports")

Limit to 5-10 most important concepts.""",
)


# ============================================================================
# Text Summarization Prompts
# ============================================================================

SUMMARIZATION = PromptTemplate(
    system="""You are a historian creating concise summaries of German newspaper articles.
Preserve key facts, dates, and names while maintaining historical accuracy.""",
    user="""Summarize the following German newspaper text.

Text:
{text}

Return a JSON object with:
- summary: Brief summary ({length} words max)
- key_points: List of 3-5 key points
- historical_context: One sentence about historical significance (if applicable)

Maintain factual accuracy and preserve important names/dates.""",
)


# ============================================================================
# Quality Assessment Prompts
# ============================================================================

TEXT_QUALITY_ASSESSMENT = PromptTemplate(
    system="""You are an expert assessing OCR quality and text coherence in historical documents.""",
    user="""Assess the quality and coherence of this OCR-scanned German newspaper text.

Text:
{text}

Return a JSON object with:
- quality_score: Overall quality 0.0-1.0
- ocr_errors: Estimated OCR error level ("low", "medium", "high")
- coherence: Text coherence level ("coherent", "partial", "fragmented")
- issues: List of detected issues (e.g., "missing words", "garbled characters", "incomplete sentences")
- readable: Boolean - is the text readable/usable for analysis?

Consider typical OCR errors in Fraktur/Gothic script.""",
)


# ============================================================================
# Helper Functions
# ============================================================================


def get_prompt(name: str) -> PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: Prompt template name (e.g., "entity_extraction", "topic_classification").

    Returns:
        PromptTemplate instance.

    Raises:
        KeyError: If prompt name not found.
    """
    prompts = {
        "entity_extraction": ENTITY_EXTRACTION,
        "topic_classification": TOPIC_CLASSIFICATION,
        "topic_generation": TOPIC_GENERATION,
        "emotion_analysis": EMOTION_ANALYSIS,
        "concept_extraction": CONCEPT_EXTRACTION,
        "summarization": SUMMARIZATION,
        "text_quality": TEXT_QUALITY_ASSESSMENT,
    }

    if name not in prompts:
        available = ", ".join(prompts.keys())
        raise KeyError(f"Unknown prompt '{name}'. Available: {available}")

    return prompts[name]


def list_prompts() -> List[str]:
    """
    List all available prompt template names.

    Returns:
        List of prompt names.
    """
    return [
        "entity_extraction",
        "topic_classification",
        "topic_generation",
        "emotion_analysis",
        "concept_extraction",
        "summarization",
        "text_quality",
    ]
