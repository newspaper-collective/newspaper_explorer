"""
Prompt template for concept and theme extraction.
"""

from newspaper_explorer.llm.prompts.base import PromptTemplate


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
