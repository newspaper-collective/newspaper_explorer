"""
Utility functions for topic modeling CLI commands.

Provides common helper functions for parsing arguments, formatting output,
and orchestrating topic modeling workflows.
"""

from pathlib import Path
from typing import Optional

import polars as pl


def parse_group_by(group_by: Optional[str]) -> Optional[list[str]]:
    """
    Parse comma-separated grouping columns.

    Args:
        group_by: Comma-separated column names or None

    Returns:
        List of column names or None

    Example:
        >>> parse_group_by("year,month")
        ['year', 'month']
        >>> parse_group_by(None)
        None
    """
    if not group_by:
        return None
    return [g.strip() for g in group_by.split(",")]


def parse_stopwords(stopwords: Optional[str]) -> Optional[list[str]]:
    """
    Parse comma-separated custom stopwords.

    Args:
        stopwords: Comma-separated stopwords or None

    Returns:
        List of stopwords or None

    Example:
        >>> parse_stopwords("der,die,das")
        ['der', 'die', 'das']
        >>> parse_stopwords(None)
        None
    """
    if not stopwords:
        return None
    return [w.strip() for w in stopwords.split(",")]


def format_topic_terms(
    topic_id: int, terms: list[str], scores: list[float], max_terms: int = 10
) -> str:
    """
    Format topic terms with scores for display.

    Args:
        topic_id: Topic identifier
        terms: List of topic terms
        scores: List of term scores
        max_terms: Maximum terms to display

    Returns:
        Formatted string

    Example:
        >>> format_topic_terms(0, ['word1', 'word2'], [0.5, 0.3])
        'Topic 0: word1(0.500), word2(0.300)'
    """
    terms_str = ", ".join(
        [f"{term}({score:.3f})" for term, score in zip(terms[:max_terms], scores[:max_terms])]
    )
    return f"Topic {topic_id}: {terms_str}"


def format_document_topics(
    doc_id: str,
    topics: list[int],
    topic_probs: list[float],
    topic_terms: list[str],
    max_topics: int = 3,
    max_terms: int = 5,
) -> str:
    """
    Format document topic assignments for display.

    Args:
        doc_id: Document identifier
        topics: List of topic IDs
        topic_probs: List of topic probabilities
        topic_terms: List of topic terms
        max_topics: Maximum topics to show
        max_terms: Maximum terms to show

    Returns:
        Formatted string
    """
    topic_info = ", ".join(
        [f"T{t}({p:.2f})" for t, p in zip(topics[:max_topics], topic_probs[:max_topics])]
    )
    terms_str = ", ".join(topic_terms[:max_terms])
    return f"Document {doc_id}\n  Main topics: {topic_info}\n  Topic terms: {terms_str}"


def merge_yearly_topics(
    source: str,
    start_year: int,
    end_year: int,
    results_dir: Path,
    output_dir: Path,
    top_n: int = 10,
) -> dict:
    """
    Merge and analyze yearly BERTopic results.

    Args:
        source: Source name
        start_year: First year to process
        end_year: Last year to process
        results_dir: Directory containing yearly topic files
        output_dir: Directory for output files
        top_n: Number of top topics per year to analyze

    Returns:
        Dictionary with:
            - yearly_files: List of (year, file_path) tuples
            - missing_years: List of missing years
            - combined: Combined DataFrame
            - topic_evolution: Topic evolution DataFrame
            - output_files: Dict of output file paths

    Raises:
        FileNotFoundError: If no yearly results found
    """
    # Load yearly files
    yearly_files = []
    missing_years = []

    for year in range(start_year, end_year + 1):
        file_path = results_dir / f"topics_{year}.parquet"
        if file_path.exists():
            yearly_files.append((year, file_path))
        else:
            missing_years.append(year)

    if not yearly_files:
        raise FileNotFoundError(f"No yearly results found in {results_dir}")

    # Load and combine
    dfs = []
    for year, file_path in yearly_files:
        df = pl.read_parquet(file_path)
        df = df.with_columns(pl.lit(year).alias("year"))
        dfs.append(df)

    combined = pl.concat(dfs)

    # Analyze topic evolution
    df_exploded = combined.select(["year", "doc_id", "topic_terms", "scores"]).explode(
        ["topic_terms", "scores"]
    )

    topic_evolution = (
        df_exploded.group_by(["year", "topic_terms"])
        .agg(
            [
                pl.count().alias("frequency"),
                pl.col("scores").mean().alias("avg_score"),
                pl.col("scores").std().alias("std_score"),
            ]
        )
        .sort(["year", "frequency"], descending=[False, True])
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_file = output_dir / f"{source}_topics_all_years.parquet"
    combined.write_parquet(combined_file)

    evolution_file = output_dir / f"{source}_topic_evolution.parquet"
    topic_evolution.write_parquet(evolution_file)

    evolution_csv = output_dir / f"{source}_topic_evolution.csv"
    topic_evolution.write_csv(evolution_csv)

    return {
        "yearly_files": yearly_files,
        "missing_years": missing_years,
        "combined": combined,
        "topic_evolution": topic_evolution,
        "output_files": {
            "combined": combined_file,
            "evolution": evolution_file,
            "evolution_csv": evolution_csv,
        },
    }


def generate_topic_visualizations(
    topic_evolution: pl.DataFrame,
    source: str,
    output_dir: Path,
    top_n_terms: int = 20,
) -> dict[str, Path]:
    """
    Generate topic evolution visualizations.

    Args:
        topic_evolution: Topic evolution DataFrame
        source: Source name
        output_dir: Directory for output files
        top_n_terms: Number of top terms to visualize

    Returns:
        Dictionary with paths to generated visualization files

    Raises:
        ImportError: If matplotlib/seaborn not installed
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get top terms overall
    top_overall = (
        topic_evolution.group_by("topic_terms")
        .agg(pl.col("frequency").sum().alias("total_frequency"))
        .sort("total_frequency", descending=True)
        .head(top_n_terms)
    )
    top_terms = top_overall["topic_terms"].to_list()

    df_filtered = topic_evolution.filter(pl.col("topic_terms").is_in(top_terms))
    df_pd = df_filtered.to_pandas()

    output_files = {}

    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    pivot = df_pd.pivot(index="topic_terms", columns="year", values="frequency").fillna(0)
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        cbar_kws={"label": "Document Frequency"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Topic Evolution - {source.upper()}", fontsize=16, pad=20)
    plt.tight_layout()

    heatmap_file = output_dir / f"{source}_topic_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    output_files["heatmap"] = heatmap_file
    plt.close()

    # Line plot
    fig, ax = plt.subplots(figsize=(14, 8))
    for term in top_terms[:10]:
        term_data = df_pd[df_pd["topic_terms"] == term]
        ax.plot(term_data["year"], term_data["frequency"], marker="o", label=term)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Document Frequency", fontsize=12)
    ax.set_title(f"Top 10 Topics Over Time - {source.upper()}", fontsize=16, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="both", alpha=0.3)
    plt.tight_layout()

    trends_file = output_dir / f"{source}_topic_trends.png"
    plt.savefig(trends_file, dpi=300, bbox_inches="tight")
    output_files["trends"] = trends_file
    plt.close()

    return output_files
