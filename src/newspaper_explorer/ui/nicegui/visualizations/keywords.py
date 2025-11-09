"""
Keyword-specific visualizations for the NiceGUI interface

Provides visualization functions for keyword extraction results,
including frequency distributions, word clouds, and temporal analysis.
"""

import io
from typing import Optional, List, Tuple

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


def create_keyword_frequency_chart(df: pl.DataFrame, top_n: int = 20) -> Optional[go.Figure]:
    """
    Create a bar chart showing the most frequent keywords across all documents

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)
        top_n: Number of top keywords to display

    Returns:
        Plotly figure or None if no data
    """
    if df is None or len(df) == 0:
        return None

    # Explode the keywords list to get individual keyword rows
    df_exploded = df.explode(["keywords", "scores"])

    # Filter out empty keywords
    df_exploded = df_exploded.filter(
        (pl.col("keywords").is_not_null()) & (pl.col("keywords") != "")
    )

    if len(df_exploded) == 0:
        return None

    # Count keyword occurrences
    keyword_counts = (
        df_exploded.group_by("keywords")
        .agg(
            pl.count().alias("count"),
            pl.col("scores").mean().alias("avg_score"),
        )
        .sort("count", descending=True)
        .head(top_n)
    )

    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=keyword_counts["count"],
                y=keyword_counts["keywords"],
                orientation="h",
                marker=dict(
                    color=keyword_counts["avg_score"],
                    colorscale="Viridis",
                    colorbar=dict(title="Avg Score"),
                    showscale=True,
                ),
                text=keyword_counts["count"],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>"
                + "Count: %{x}<br>"
                + "Avg Score: %{marker.color:.3f}<br>"
                + "<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        xaxis_title="Frequency",
        yaxis_title="Keyword",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 25),  # Dynamic height based on number of keywords
        margin=dict(l=200, r=50, t=50, b=50),
        showlegend=False,
    )

    return fig


def create_keyword_score_distribution(df: pl.DataFrame) -> Optional[go.Figure]:
    """
    Create a histogram showing the distribution of keyword scores

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)

    Returns:
        Plotly figure or None if no data
    """
    if df is None or len(df) == 0:
        return None

    # Explode the scores list
    df_exploded = df.explode(["keywords", "scores"])

    # Filter out null scores
    df_exploded = df_exploded.filter(pl.col("scores").is_not_null())

    if len(df_exploded) == 0:
        return None

    # Create histogram
    fig = px.histogram(
        df_exploded.to_pandas(),
        x="scores",
        nbins=50,
        labels={"scores": "Keyword Score", "count": "Frequency"},
        title="Distribution of Keyword Relevance Scores",
    )

    fig.update_traces(marker=dict(color="#2E5EFF", line=dict(color="white", width=1)))

    fig.update_layout(
        xaxis_title="Keyword Score",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
    )

    return fig


def create_keywords_per_document_chart(df: pl.DataFrame) -> Optional[go.Figure]:
    """
    Create a histogram showing the distribution of keyword counts per document

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)

    Returns:
        Plotly figure or None if no data
    """
    if df is None or len(df) == 0:
        return None

    # Calculate keyword count per document
    keyword_counts = df.with_columns(pl.col("keywords").list.len().alias("keyword_count")).filter(
        pl.col("keyword_count") > 0
    )

    if len(keyword_counts) == 0:
        return None

    # Create histogram
    fig = px.histogram(
        keyword_counts.to_pandas(),
        x="keyword_count",
        nbins=30,
        labels={"keyword_count": "Keywords per Document", "count": "Frequency"},
        title="Distribution of Keywords per Document",
    )

    fig.update_traces(marker=dict(color="#00E676", line=dict(color="white", width=1)))

    fig.update_layout(
        xaxis_title="Number of Keywords",
        yaxis_title="Number of Documents",
        showlegend=False,
        height=400,
    )

    return fig


def create_keyword_wordcloud(
    df: pl.DataFrame,
    max_words: int = 200,
    width: int = 800,
    height: int = 400,
) -> Optional[bytes]:
    """
    Create a word cloud visualization from keywords weighted by their scores

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)
        max_words: Maximum number of words to include
        width: Width of the word cloud image
        height: Height of the word cloud image

    Returns:
        PNG image bytes or None if no data
    """
    if df is None or len(df) == 0:
        return None

    # Explode the keywords and scores
    df_exploded = df.explode(["keywords", "scores"])

    # Filter out empty keywords
    df_exploded = df_exploded.filter(
        (pl.col("keywords").is_not_null()) & (pl.col("keywords") != "")
    )

    if len(df_exploded) == 0:
        return None

    # Aggregate keywords with their average scores as weights
    keyword_weights = (
        df_exploded.group_by("keywords")
        .agg(
            pl.col("scores").mean().alias("weight"),
            pl.count().alias("frequency"),
        )
        .with_columns(
            # Combine score and frequency for better weighting
            (pl.col("weight") * pl.col("frequency").log1p()).alias("combined_weight")
        )
        .sort("combined_weight", descending=True)
        .head(max_words)
    )

    # Create word frequency dictionary
    word_freq = dict(
        zip(
            keyword_weights["keywords"].to_list(),
            keyword_weights["combined_weight"].to_list(),
        )
    )

    if not word_freq:
        return None

    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap="viridis",
        min_font_size=10,
    ).generate_from_frequencies(word_freq)

    # Convert to PNG bytes
    img_buffer = io.BytesIO()
    wordcloud.to_image().save(img_buffer, format="PNG")
    img_buffer.seek(0)

    return img_buffer.getvalue()


def get_keyword_statistics(df: pl.DataFrame) -> Optional[dict]:
    """
    Calculate overall statistics about keywords in the dataset

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)

    Returns:
        Dictionary with statistics or None if no data
    """
    if df is None or len(df) == 0:
        return None

    # Count documents with keywords
    docs_with_keywords = df.filter(pl.col("keywords").list.len() > 0)

    if len(docs_with_keywords) == 0:
        return {
            "total_documents": len(df),
            "documents_with_keywords": 0,
            "total_keywords": 0,
            "unique_keywords": 0,
            "avg_keywords_per_doc": 0.0,
            "avg_score": 0.0,
        }

    # Explode for detailed stats
    df_exploded = docs_with_keywords.explode(["keywords", "scores"])
    df_exploded = df_exploded.filter(
        (pl.col("keywords").is_not_null()) & (pl.col("keywords") != "")
    )

    total_keywords = len(df_exploded)
    unique_keywords = df_exploded["keywords"].n_unique()

    # Calculate keyword counts per document
    keywords_per_doc = docs_with_keywords.with_columns(pl.col("keywords").list.len().alias("count"))

    stats = {
        "total_documents": len(df),
        "documents_with_keywords": len(docs_with_keywords),
        "total_keywords": total_keywords,
        "unique_keywords": unique_keywords,
        "avg_keywords_per_doc": keywords_per_doc["count"].mean(),
        "avg_score": df_exploded["scores"].mean() if len(df_exploded) > 0 else 0.0,
        "min_score": df_exploded["scores"].min() if len(df_exploded) > 0 else 0.0,
        "max_score": df_exploded["scores"].max() if len(df_exploded) > 0 else 0.0,
    }

    return stats


def get_top_keywords(df: pl.DataFrame, n: int = 50) -> Optional[pl.DataFrame]:
    """
    Get the top N most frequent keywords with their statistics

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)
        n: Number of top keywords to return

    Returns:
        DataFrame with columns: keyword, count, avg_score, min_score, max_score
    """
    if df is None or len(df) == 0:
        return None

    # Explode the keywords list
    df_exploded = df.explode(["keywords", "scores"])

    # Filter out empty keywords
    df_exploded = df_exploded.filter(
        (pl.col("keywords").is_not_null()) & (pl.col("keywords") != "")
    )

    if len(df_exploded) == 0:
        return None

    # Aggregate by keyword
    top_keywords = (
        df_exploded.group_by("keywords")
        .agg(
            pl.count().alias("count"),
            pl.col("scores").mean().alias("avg_score"),
            pl.col("scores").min().alias("min_score"),
            pl.col("scores").max().alias("max_score"),
        )
        .sort("count", descending=True)
        .head(n)
    )

    return top_keywords


def extract_date_from_doc_id(doc_id: str) -> Optional[str]:
    """
    Extract date from doc_id string in various formats

    Handles:
    - ISO datetime: "1901-01-08 00:00:00"
    - Text block IDs: "3074409-X_1901-01-08_006_1_001_r_14_1"

    Args:
        doc_id: Document ID string

    Returns:
        Date string in YYYY-MM-DD format or None
    """
    import re

    # Pattern 1: ISO datetime (e.g., "1901-01-08 00:00:00")
    if " " in doc_id and ":" in doc_id:
        return doc_id.split(" ")[0]

    # Pattern 2: Text block ID with date (e.g., "3074409-X_1901-01-08_...")
    match = re.search(r"(\d{4}-\d{2}-\d{2})", doc_id)
    if match:
        return match.group(1)

    return None


def create_keyword_timeline(
    df: pl.DataFrame, selected_keywords: Optional[List[str]] = None, top_n: int = 5
) -> Optional[go.Figure]:
    """
    Create a timeline showing keyword frequency over time

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)
        selected_keywords: List of keywords to display. If None, shows top N keywords
        top_n: Number of top keywords to show if selected_keywords is None

    Returns:
        Plotly figure or None if no data
    """
    if df is None or len(df) == 0:
        return None

    # Add date column by extracting from doc_id
    df_with_date = df.with_columns(
        pl.col("doc_id").map_elements(extract_date_from_doc_id, return_dtype=pl.Utf8).alias("date")
    )

    # Filter out rows without dates
    df_with_date = df_with_date.filter(pl.col("date").is_not_null())

    if len(df_with_date) == 0:
        return None

    # Convert date strings to datetime
    try:
        df_with_date = df_with_date.with_columns(pl.col("date").str.to_date().alias("date"))
    except Exception:
        return None

    # Explode keywords
    df_exploded = df_with_date.explode(["keywords", "scores"])
    df_exploded = df_exploded.filter(
        (pl.col("keywords").is_not_null()) & (pl.col("keywords") != "")
    )

    if len(df_exploded) == 0:
        return None

    # If no keywords selected, get top N
    if not selected_keywords:
        top_kw = get_top_keywords(df, n=top_n)
        if top_kw is None:
            return None
        selected_keywords = top_kw["keywords"].to_list()

    # Filter to selected keywords
    df_filtered = df_exploded.filter(pl.col("keywords").is_in(selected_keywords))

    if len(df_filtered) == 0:
        return None

    # Aggregate by date and keyword
    timeline_data = (
        df_filtered.group_by(["date", "keywords"]).agg(pl.count().alias("count")).sort("date")
    )

    # Create line chart
    fig = px.line(
        timeline_data.to_pandas(),
        x="date",
        y="count",
        color="keywords",
        markers=True,
        labels={"count": "Frequency", "date": "Date", "keywords": "Keyword"},
        title="Keyword Frequency Over Time",
    )

    fig.update_layout(
        hovermode="x unified",
        height=500,
        xaxis_title="Date",
        yaxis_title="Frequency",
        legend_title="Keywords",
    )

    return fig


def get_documents_for_keyword(
    df: pl.DataFrame, keyword: str, limit: int = 50
) -> Optional[pl.DataFrame]:
    """
    Get all documents containing a specific keyword

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)
        keyword: The keyword to search for
        limit: Maximum number of documents to return

    Returns:
        DataFrame with columns: doc_id, score, date (if extractable)
    """
    if df is None or len(df) == 0:
        return None

    # Explode to get keyword-document pairs
    df_exploded = df.explode(["keywords", "scores"])

    # Filter for the specific keyword
    df_keyword = df_exploded.filter(pl.col("keywords") == keyword)

    if len(df_keyword) == 0:
        return None

    # Select relevant columns and sort by score
    result = (
        df_keyword.select(["doc_id", "scores"])
        .rename({"scores": "score"})
        .sort("score", descending=True)
        .head(limit)
    )

    # Try to extract dates
    result = result.with_columns(
        pl.col("doc_id").map_elements(extract_date_from_doc_id, return_dtype=pl.Utf8).alias("date")
    )

    # Try to convert to datetime
    try:
        result = result.with_columns(pl.col("date").str.to_date().alias("date"))
    except Exception:
        pass

    return result


def get_keyword_cooccurrence(
    df: pl.DataFrame, keyword: str, top_n: int = 10
) -> Optional[pl.DataFrame]:
    """
    Get keywords that frequently co-occur with a given keyword

    Args:
        df: Polars DataFrame with columns: doc_id, keywords (list), scores (list)
        keyword: The keyword to find co-occurrences for
        top_n: Number of top co-occurring keywords to return

    Returns:
        DataFrame with columns: cooccurring_keyword, count
    """
    if df is None or len(df) == 0:
        return None

    # Find documents containing the target keyword
    docs_with_keyword = df.filter(pl.col("keywords").list.contains(keyword))

    if len(docs_with_keyword) == 0:
        return None

    # Explode all keywords from these documents
    all_keywords = docs_with_keyword.explode("keywords")

    # Filter out the original keyword and empty values
    all_keywords = all_keywords.filter(
        (pl.col("keywords") != keyword)
        & (pl.col("keywords").is_not_null())
        & (pl.col("keywords") != "")
    )

    if len(all_keywords) == 0:
        return None

    # Count co-occurrences
    cooccurrence = (
        all_keywords.group_by("keywords")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(top_n)
        .rename({"keywords": "cooccurring_keyword"})
    )

    return cooccurrence
