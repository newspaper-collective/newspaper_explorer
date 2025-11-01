"""
Prompt templates for topic analysis and classification.
"""

from newspaper_explorer.llm.prompts.base import PromptTemplate


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
