"""
Base prompt template class for LLM interactions.
"""

from typing import Any, Dict, Optional


class PromptTemplate:
    """
    Reusable prompt template with variable substitution and metadata support.

    Templates can include placeholders for:
    - {text} - The main text content to analyze
    - {source} - Source name (e.g., "Der Tag")
    - {date} - Publication date
    - {newspaper_title} - Full newspaper title
    - {year_volume} - Year and volume information
    - {page_number} - Page number in the issue
    - Any other custom metadata fields

    Example:
        ```python
        template = PromptTemplate(
            system="You are a historian analyzing {source}.",
            user="Analyze this text from {date}: {text}"
        )

        # Basic usage
        prompt = template.format(text="Berlin 1920", date="1920-01-15")

        # With metadata dict
        metadata = {"source": "Der Tag", "date": "1920-01-15"}
        prompt = template.format(text="Berlin 1920", metadata=metadata)
        ```
    """

    def __init__(self, system: str = "", user: str = "", include_metadata: bool = False):
        """
        Initialize prompt template.

        Args:
            system: System message template (model behavior instructions).
            user: User message template (task-specific prompt).
            include_metadata: If True, automatically append metadata context to prompt.
        """
        self.system = system
        self.user = user
        self.include_metadata = include_metadata

    def format(self, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, str]:
        """
        Format template with provided variables and optional metadata.

        Args:
            metadata: Optional dict with source metadata (source, date, newspaper_title, etc.)
            **kwargs: Direct variables to substitute in templates (e.g., text="...")

        Returns:
            Dict with 'system' and 'user' keys containing formatted prompts.

        Example:
            ```python
            # Direct kwargs
            prompt = template.format(text="...", source="Der Tag")

            # With metadata dict
            metadata = {"source": "Der Tag", "date": "1920-01-15"}
            prompt = template.format(text="...", metadata=metadata)

            # Both (kwargs override metadata)
            prompt = template.format(text="...", metadata=metadata, source="Override")
            ```
        """
        # Merge metadata and kwargs (kwargs take precedence)
        format_vars = {}
        if metadata:
            format_vars.update(metadata)
        format_vars.update(kwargs)

        # Format base prompts
        user_prompt = self.user.format(**format_vars)

        # Optionally append metadata context
        if self.include_metadata and metadata:
            metadata_context = self._format_metadata_context(metadata)
            if metadata_context:
                user_prompt = f"{user_prompt}\n\n{metadata_context}"

        return {
            "system": self.system.format(**format_vars) if self.system else "",
            "user": user_prompt,
        }

    def _format_metadata_context(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata into a context string.

        Args:
            metadata: Metadata dictionary

        Returns:
            Formatted metadata context string
        """
        context_parts = []

        if "source" in metadata or "newspaper_title" in metadata:
            context_parts.append(
                f"Source: {metadata.get('newspaper_title', metadata.get('source', 'Unknown'))}"
            )

        if "date" in metadata:
            context_parts.append(f"Publication Date: {metadata['date']}")

        if "year_volume" in metadata:
            context_parts.append(f"Volume: {metadata['year_volume']}")

        if "page_number" in metadata:
            context_parts.append(f"Page: {metadata['page_number']}")

        if context_parts:
            return "Context:\n" + "\n".join(f"- {part}" for part in context_parts)

        return ""
