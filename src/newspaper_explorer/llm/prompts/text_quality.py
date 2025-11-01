"""
Prompt template for text quality assessment.
"""

from newspaper_explorer.llm.prompts.base import PromptTemplate


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
