"""
Prompt template for keyword and keyphrase extraction.
"""

from newspaper_explorer.llm.prompts.base import PromptTemplate


KEYWORD_EXTRACTION = PromptTemplate(
    system="""You are an expert at keyword and keyphrase extraction from historical German newspaper texts.
Your task is to identify the most important and representative keywords/keyphrases that capture:
- Main topics and themes discussed
- Key concepts and ideas
- Important entities mentioned
- Significant events or actions

Consider the historical context (early 20th century German newspapers) and preserve original spelling.
The text is from {source} published around {date}.""",
    user="""Extract keywords and keyphrases from the following German newspaper text.

Text:
{text}

Return a JSON object with:
- keywords: List of 10-15 most important keywords/keyphrases from the text
- scores: List of confidence scores (0.0-1.0) for each keyword, indicating how representative/important it is

Guidelines:
- Include both single words (e.g., "Berlin", "Krieg") and multi-word phrases (e.g., "Deutsche Reich", "Reichskanzler")
- Focus on content words (nouns, proper nouns, key verbs) not stopwords
- Preserve historical spelling and terminology
- Prioritize keywords that capture the main topic and key information
- Assign higher scores (0.7-1.0) to central topics, lower scores (0.3-0.6) to supporting concepts
- Order by importance (most important first)

Return empty lists if the text contains no meaningful keywords.""",
    include_metadata=True,
)
