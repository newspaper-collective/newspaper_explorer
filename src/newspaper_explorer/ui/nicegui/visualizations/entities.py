"""
Entity-specific visualizations for the NiceGUI interface

Provides visualization functions specific to entity analysis,
including timeline charts, type distributions, and word clouds.
"""

import io
from typing import Optional, Dict

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Shared color palette for consistent visualization across all charts
# Using vivid, highly saturated colors for better visibility
ENTITY_COLORS = [
    "#2E5EFF",  # Vibrant Blue
    "#FF3333",  # Vivid Red
    "#00E676",  # Bright Green
    "#9C27FF",  # Vivid Purple
    "#FF9100",  # Bright Orange
    "#00E5FF",  # Bright Cyan
    "#FF1744",  # Bright Pink/Red
    "#76FF03",  # Neon Green
    "#E040FB",  # Bright Magenta
    "#FFEA00",  # Bright Yellow
]

# Global color mapping for entity types (populated when data is loaded)
_ENTITY_TYPE_COLOR_MAP: Dict[str, str] = {}


def assign_entity_type_colors(df: pl.DataFrame) -> Dict[str, str]:
    """
    Assign consistent colors to all entity types found in the dataset.
    This should be called once when entity data is loaded.

    Args:
        df: Polars DataFrame with entity data containing entity_type column

    Returns:
        Dictionary mapping entity type names to color hex codes
    """
    global _ENTITY_TYPE_COLOR_MAP

    if df is None or len(df) == 0:
        _ENTITY_TYPE_COLOR_MAP = {}
        return {}

    if "entity_type" not in df.columns:
        _ENTITY_TYPE_COLOR_MAP = {}
        return {}

    # Get unique entity types and sort alphabetically for consistency
    entity_types = sorted(df["entity_type"].unique().drop_nulls().to_list())

    # Assign colors from the palette
    _ENTITY_TYPE_COLOR_MAP = {
        entity_type: ENTITY_COLORS[i % len(ENTITY_COLORS)]
        for i, entity_type in enumerate(entity_types)
    }

    return _ENTITY_TYPE_COLOR_MAP


def get_entity_type_color_map() -> Dict[str, str]:
    """
    Get the current global entity type to color mapping.

    Returns:
        Dictionary mapping entity type names to color hex codes
    """
    return _ENTITY_TYPE_COLOR_MAP.copy()


def get_entity_type_color(entity_type: str) -> Optional[str]:
    """
    Get the assigned color for a specific entity type.

    Args:
        entity_type: The entity type name

    Returns:
        Color hex code or None if type not found
    """
    return _ENTITY_TYPE_COLOR_MAP.get(entity_type)


def create_entity_timeline_chart(df: pl.DataFrame) -> Optional[go.Figure]:
    """
    Create a bar chart showing entity mentions over time by type

    Args:
        df: Polars DataFrame with columns: date, entity_type

    Returns:
        Plotly figure or None if no data
    """
    if df is None or len(df) == 0:
        return None

    if "entity_type" not in df.columns or "date" not in df.columns:
        return None

    # Group by date and entity type
    entity_counts = df.group_by(["date", "entity_type"]).agg(pl.count().alias("Count")).sort("date")

    # Convert to pandas for plotly (it handles pandas better)
    entity_counts_pd = entity_counts.to_pandas()

    if len(entity_counts_pd) == 0:
        return None

    # Use global color map if available, otherwise create one
    color_map = (
        _ENTITY_TYPE_COLOR_MAP
        if _ENTITY_TYPE_COLOR_MAP
        else {
            entity_type: ENTITY_COLORS[i % len(ENTITY_COLORS)]
            for i, entity_type in enumerate(sorted(entity_counts_pd["entity_type"].unique()))
        }
    )

    # Create bar chart with explicit color mapping
    fig = px.bar(
        entity_counts_pd,
        x="date",
        y="Count",
        color="entity_type",
        labels={"date": "Date", "entity_type": "Entity Type"},
        barmode="group",  # Group bars for better clarity
        color_discrete_map=color_map,
    )

    # Update bar traces to remove any transparency and enhance visibility
    fig.update_traces(
        marker=dict(
            line=dict(width=0),  # Remove bar borders for cleaner look
            opacity=1.0,  # Full opacity - no transparency
        )
    )

    # Customize layout with white/light background
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20, b=60, l=60, r=20),
        height=400,
        plot_bgcolor="white",  # White background for plot area
        paper_bgcolor="white",  # White background for entire figure
    )

    # Update axes styling for better visibility
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig


def create_entity_type_pie_chart(df: pl.DataFrame) -> Optional[go.Figure]:
    """
    Create a pie chart showing distribution of entity types

    Args:
        df: Polars DataFrame with columns: entity_type

    Returns:
        Plotly figure or None if no data
    """
    if df is None or len(df) == 0:
        return None

    if "entity_type" not in df.columns:
        return None

    # Count by type
    type_counts = (
        df.group_by("entity_type").agg(pl.count().alias("Count")).sort("Count", descending=True)
    )

    if len(type_counts) == 0:
        return None

    # Convert to pandas
    type_counts_pd = type_counts.to_pandas()

    # Use global color map if available, otherwise create one
    # The global map uses the exact type names from the data
    color_map = (
        _ENTITY_TYPE_COLOR_MAP
        if _ENTITY_TYPE_COLOR_MAP
        else {
            entity_type: ENTITY_COLORS[i % len(ENTITY_COLORS)]
            for i, entity_type in enumerate(sorted(type_counts_pd["entity_type"].unique()))
        }
    )

    # Create pie chart with explicit color mapping
    fig = px.pie(
        type_counts_pd,
        values="Count",
        names="entity_type",
        color="entity_type",
        color_discrete_map=color_map,
    )

    # Customize layout
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        showlegend=True,
        margin=dict(t=20, b=20, l=20, r=20),  # Reduce margins
        height=500,  # Set fixed height to match the list panel
    )

    return fig


def get_top_entities(df: pl.DataFrame, limit: int = 10) -> pl.DataFrame:
    """
    Get top entities by frequency

    Args:
        df: Polars DataFrame with columns: entity_text, entity_type
        limit: Number of top entities to return

    Returns:
        Polars DataFrame with columns: Entity, entity_type, Count
    """
    if df is None or len(df) == 0:
        return pl.DataFrame({"Entity": [], "entity_type": [], "Count": []})

    if "entity_text" not in df.columns or "entity_type" not in df.columns:
        return pl.DataFrame({"Entity": [], "entity_type": [], "Count": []})

    # Group by entity and type, count occurrences
    top_entities = (
        df.group_by(["entity_text", "entity_type"])
        .agg(pl.count().alias("Count"))
        .sort("Count", descending=True)
        .head(limit)
    )

    # Rename columns to standard format for display (keep entity_type as-is for compatibility)
    return top_entities.rename({"entity_text": "Entity"})


def create_wordcloud_image(
    df: pl.DataFrame,
    entity_type: str,
    max_words: int = 200,
    width: int = 1600,
    height: int = 800,
    colormap: str = "viridis",
) -> Optional[bytes]:
    """
    Create a word cloud image from entity frequencies

    Args:
        df: Polars DataFrame with columns: entity_text, entity_type
        entity_type: Type of entities to include in word cloud
        max_words: Maximum number of words in the cloud
        width: Width of the image
        height: Height of the image
        colormap: Matplotlib colormap to use

    Returns:
        PNG image as bytes or None if no data
    """
    if df is None or len(df) == 0:
        return None

    if "entity_text" not in df.columns or "entity_type" not in df.columns:
        return None

    # Filter by entity type (case-insensitive)
    filtered = df.filter(pl.col("entity_type").str.to_lowercase() == entity_type.lower())

    if len(filtered) == 0:
        return None

    # Count entity frequencies
    freqs = (
        filtered.group_by("entity_text")
        .agg(pl.count().alias("count"))
        .to_pandas()
        .set_index("entity_text")["count"]
        .to_dict()
    )

    if not freqs:
        return None

    try:
        # Generate word cloud
        wc = WordCloud(
            width=width,
            height=height,
            background_color="white",
            max_words=max_words,
            colormap=colormap,
        )
        wc.generate_from_frequencies(freqs)

        # Convert to PNG bytes
        img = wc.to_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None


def get_available_entity_types(df: pl.DataFrame) -> list:
    """
    Get list of unique entity types from DataFrame

    Args:
        df: Polars DataFrame with columns: entity_type

    Returns:
        Sorted list of entity types
    """
    if df is None or len(df) == 0:
        return []

    if "entity_type" not in df.columns:
        return []

    # Get unique types and sort
    types = df["entity_type"].unique().drop_nulls().sort().to_list()
    return types
