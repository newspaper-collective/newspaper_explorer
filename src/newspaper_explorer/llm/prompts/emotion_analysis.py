"""
Prompt template for emotion and sentiment analysis.
"""

from newspaper_explorer.llm.prompts.base import PromptTemplate


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
