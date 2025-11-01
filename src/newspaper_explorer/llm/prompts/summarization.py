"""
Prompt template for text summarization.
"""

from newspaper_explorer.llm.prompts.base import PromptTemplate


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
