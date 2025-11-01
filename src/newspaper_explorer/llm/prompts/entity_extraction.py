"""
Prompt template for named entity extraction.
"""

from newspaper_explorer.llm.prompts.base import PromptTemplate


ENTITY_EXTRACTION = PromptTemplate(
    system="""You are an expert at named entity recognition in historical German newspaper texts.
Extract entities accurately, preserving original spelling and historical names.
Consider historical context (e.g., place names may differ from modern names).
The text is from {source} published around {date}.""",
    user="""Extract named entities from the following German newspaper text.

Text:
{text}

Return a JSON object with these fields:
- persons: List of person names (e.g., "Kaiser Wilhelm II", "Otto von Bismarck")
- locations: List of places (e.g., "Berlin", "Deutsches Reich", "Wien")
- organizations: List of organizations (e.g., "Reichstag", "Sozialdemokratische Partei")

Use empty lists if no entities found. Preserve historical spelling.""",
    include_metadata=True,
)
