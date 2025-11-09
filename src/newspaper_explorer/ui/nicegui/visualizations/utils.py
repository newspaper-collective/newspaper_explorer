"""
General visualization utilities for the NiceGUI interface

Provides reusable helper functions for creating charts and visualizations
across different analysis modules.
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt


def get_wordcloud_class() -> type:
    """
    Get WordCloud class

    Returns:
        WordCloud class
    """
    return WordCloud


def get_matplotlib():
    """
    Get matplotlib.pyplot module

    Returns:
        matplotlib.pyplot module
    """
    return plt
