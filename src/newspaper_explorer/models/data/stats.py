"""
Type definitions for text analysis statistics.

These TypedDicts provide proper typing for statistics returned by
text analysis functions, enabling autocomplete and type checking.
"""

from typing import TypedDict


class CharDistribution(TypedDict):
    """Character length distribution counts."""

    under_50: int
    under_100: int
    under_200: int
    under_500: int
    under_1000: int


class CharLengthStats(TypedDict):
    """Statistics from character length analysis."""

    total_rows: int
    sample_size: int
    min_chars: int
    max_chars: int
    mean_chars: float
    median_chars: int
    p90_chars: int
    p95_chars: int
    p99_chars: int
    distribution: CharDistribution
    longest_examples: list[tuple[int, str]]


class TokenDistribution(TypedDict):
    """Token length distribution counts."""

    under_50: int
    under_100: int
    under_200: int
    under_300: int
    at_max_length: int


class TokenLengthStats(TypedDict):
    """Statistics from token length analysis."""

    total_rows: int
    sample_size: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    median_tokens: int
    p90_tokens: int
    p95_tokens: int
    p99_tokens: int
    distribution: TokenDistribution
    truncated_count: int
    truncated_percent: float
    wasted_padding_percent: float
    expected_speedup: float
    longest_examples: list[tuple[int, str]]
