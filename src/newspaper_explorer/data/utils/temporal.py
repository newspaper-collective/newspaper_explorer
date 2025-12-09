"""
Temporal metadata utilities for historical newspaper analysis.

Provides era classification based on year/date columns with configurable
presets for common historical periodizations (WWI, Weimar, etc.).

Usage:
    from newspaper_explorer.data.utils.temporal import add_era_columns, ERA_PRESETS

    # Add era columns using a preset
    df = add_era_columns(df, preset="wwi")

    # Or with custom era definition
    df = add_era_columns(df, eras={
        "early": (1900, 1910),
        "late": (1911, 1920),
    })
"""

from datetime import date, datetime
import logging
from typing import Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Era Presets for 1900-1920 German Newspaper Analysis
# =============================================================================

ERA_PRESETS: dict[str, dict] = {
    "wwi": {
        "name": "World War I Periodization",
        "description": "Standard WWI periodization: pre-war, war years, post-war",
        "eras": {
            "pre_war": (None, 1913),  # until end of 1913
            "war": (1914, 1918),  # 1914-1918
            "post_war": (1919, None),  # 1919 onwards
        },
    },
    "wwi_binary": {
        "name": "WWI Binary Split",
        "description": "Simple before/after WWI start (pre/post July 1914)",
        "eras": {
            "pre": (None, 1913),
            "post": (1914, None),
        },
    },
    "mid_war_split": {
        "name": "Mid-War Split",
        "description": "Binary split at 1915/1916 boundary (initial enthusiasm vs. attrition)",
        "eras": {
            "pre": (None, 1915),
            "post": (1916, None),
        },
    },
    "war_phases": {
        "name": "War Phases",
        "description": "Detailed war phases: pre-war, early war (movement), late war (attrition), post-war",
        "eras": {
            "pre_war": (None, 1913),
            "early_war": (1914, 1915),  # War of movement, initial enthusiasm
            "late_war": (1916, 1918),  # Attrition, disillusionment
            "post_war": (1919, None),
        },
    },
    "decades": {
        "name": "Decades",
        "description": "Simple decade-based periodization",
        "eras": {
            "1900s": (1900, 1909),
            "1910s": (1910, 1919),
            "1920s": (1920, 1929),
        },
    },
    "five_year": {
        "name": "Five-Year Periods",
        "description": "Five-year periodization for finer granularity",
        "eras": {
            "1900_1904": (1900, 1904),
            "1905_1909": (1905, 1909),
            "1910_1914": (1910, 1914),
            "1915_1919": (1915, 1919),
            "1920_1924": (1920, 1924),
        },
    },
}


def list_presets() -> dict[str, str]:
    """
    List available era presets with their descriptions.

    Returns:
        Dictionary mapping preset names to descriptions.

    Example:
        >>> from newspaper_explorer.data.utils.temporal import list_presets
        >>> for name, desc in list_presets().items():
        ...     print(f"{name}: {desc}")
    """
    return {name: preset["description"] for name, preset in ERA_PRESETS.items()}


def get_preset(name: str) -> dict:
    """
    Get a specific era preset by name.

    Args:
        name: Preset name (e.g., "wwi", "war_phases")

    Returns:
        Preset dictionary with 'name', 'description', and 'eras' keys.

    Raises:
        ValueError: If preset name is not found.
    """
    if name not in ERA_PRESETS:
        available = ", ".join(ERA_PRESETS.keys())
        msg = f"Unknown preset '{name}'. Available: {available}"
        raise ValueError(msg)
    return ERA_PRESETS[name]


def classify_year(
    year: Optional[int],
    eras: dict[str, tuple[Optional[int], Optional[int]]],
) -> str:
    """
    Classify a single year into an era.

    Args:
        year: Year to classify (can be None)
        eras: Dictionary mapping era names to (start_year, end_year) tuples.
              Use None for open-ended ranges.

    Returns:
        Era name, or empty string if year is None or doesn't match any era.

    Example:
        >>> eras = {"pre": (None, 1913), "war": (1914, 1918), "post": (1919, None)}
        >>> classify_year(1912, eras)
        'pre'
        >>> classify_year(1916, eras)
        'war'
    """
    if year is None:
        return ""

    for era_name, (start, end) in eras.items():
        start_ok = start is None or year >= start
        end_ok = end is None or year <= end
        if start_ok and end_ok:
            return era_name

    return ""


def _extract_year_from_column(
    df: pl.DataFrame,
    year_column: Optional[str],
    date_column: Optional[str],
) -> pl.Expr:
    """
    Create a Polars expression to extract year from available columns.

    Tries year_column first, then extracts from date_column if needed.
    """
    # Check which columns exist
    has_year = year_column and year_column in df.columns
    has_date = date_column and date_column in df.columns

    if has_year:
        return pl.col(year_column).cast(pl.Int64)
    elif has_date:
        # Handle both datetime and date types
        col = pl.col(date_column)
        dtype = df.schema.get(date_column)
        if dtype in (pl.Date, pl.Datetime):
            return col.dt.year()
        else:
            # Try parsing as string
            return col.str.to_datetime().dt.year()
    else:
        msg = f"Neither year column '{year_column}' nor date column '{date_column}' found in DataFrame"
        raise ValueError(msg)


def add_era_columns(
    df: pl.DataFrame,
    *,
    preset: Optional[str] = None,
    eras: Optional[dict[str, tuple[Optional[int], Optional[int]]]] = None,
    year_column: str = "year",
    date_column: str = "date",
    output_column: str = "era",
) -> pl.DataFrame:
    """
    Add era classification column(s) to a DataFrame based on year/date.

    Can use a preset or custom era definition. At least one of `preset` or `eras`
    must be provided.

    Args:
        df: Input DataFrame with year or date column
        preset: Name of era preset to use (e.g., "wwi", "war_phases")
        eras: Custom era definition as dict mapping era names to (start, end) tuples.
              Use None for open-ended ranges: {"pre": (None, 1913), "post": (1914, None)}
        year_column: Name of year column (default: "year")
        date_column: Name of date/datetime column to extract year from (default: "date")
        output_column: Name for the output era column (default: "era")

    Returns:
        DataFrame with added era column.

    Raises:
        ValueError: If neither preset nor eras provided, or if required columns missing.

    Example:
        >>> from newspaper_explorer.data.utils.temporal import add_era_columns
        >>>
        >>> # Using a preset
        >>> df = add_era_columns(df, preset="wwi")
        >>>
        >>> # Using custom eras
        >>> df = add_era_columns(df, eras={
        ...     "early": (1900, 1910),
        ...     "late": (1911, 1920),
        ... })
        >>>
        >>> # Multiple era columns with different presets
        >>> df = add_era_columns(df, preset="wwi", output_column="era_wwi")
        >>> df = add_era_columns(df, preset="mid_war_split", output_column="era_binary")
    """
    # Validate inputs
    if preset is None and eras is None:
        msg = "Must provide either 'preset' or 'eras' argument"
        raise ValueError(msg)

    # Get era definitions
    if preset is not None:
        preset_config = get_preset(preset)
        era_defs = preset_config["eras"]
        logger.info(f"Using era preset '{preset}': {preset_config['name']}")
    else:
        era_defs = eras
        logger.info(f"Using custom eras: {list(era_defs.keys())}")

    # Extract year expression
    year_expr = _extract_year_from_column(df, year_column, date_column)

    # Build when/then/otherwise chain for era classification
    # Start with the first era
    era_items = list(era_defs.items())
    first_name, (first_start, first_end) = era_items[0]

    # Build condition for first era
    def _build_condition(start: Optional[int], end: Optional[int]) -> pl.Expr:
        conditions = []
        if start is not None:
            conditions.append(year_expr >= start)
        if end is not None:
            conditions.append(year_expr <= end)

        if not conditions:
            return pl.lit(True)
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return conditions[0] & conditions[1]

    # Build the expression chain
    expr = pl.when(_build_condition(first_start, first_end)).then(pl.lit(first_name))

    for era_name, (start, end) in era_items[1:]:
        expr = expr.when(_build_condition(start, end)).then(pl.lit(era_name))

    expr = expr.otherwise(pl.lit("")).alias(output_column)

    return df.with_columns(expr)


def add_multiple_era_columns(
    df: pl.DataFrame,
    presets: list[str],
    *,
    year_column: str = "year",
    date_column: str = "date",
    prefix: str = "era_",
) -> pl.DataFrame:
    """
    Add multiple era columns from different presets.

    Convenience function to add several era classifications at once.

    Args:
        df: Input DataFrame
        presets: List of preset names to apply
        year_column: Name of year column
        date_column: Name of date column
        prefix: Prefix for output column names (default: "era_")

    Returns:
        DataFrame with added era columns named "{prefix}{preset_name}".

    Example:
        >>> df = add_multiple_era_columns(df, ["wwi", "mid_war_split", "decades"])
        >>> # Adds columns: era_wwi, era_mid_war_split, era_decades
    """
    for preset_name in presets:
        output_col = f"{prefix}{preset_name}"
        df = add_era_columns(
            df,
            preset=preset_name,
            year_column=year_column,
            date_column=date_column,
            output_column=output_col,
        )
    return df


def get_era_counts(
    df: pl.DataFrame,
    era_column: str = "era",
) -> pl.DataFrame:
    """
    Get counts of rows per era.

    Args:
        df: DataFrame with era column
        era_column: Name of era column

    Returns:
        DataFrame with era and count columns, sorted by era.
    """
    return df.group_by(era_column).agg(pl.len().alias("count")).sort(era_column)


def filter_by_era(
    df: pl.DataFrame,
    era_values: Union[str, list[str]],
    era_column: str = "era",
) -> pl.DataFrame:
    """
    Filter DataFrame to specific era(s).

    Args:
        df: DataFrame with era column
        era_values: Era value(s) to keep (single string or list)
        era_column: Name of era column

    Returns:
        Filtered DataFrame.

    Example:
        >>> # Single era
        >>> war_df = filter_by_era(df, "war")
        >>> # Multiple eras
        >>> wartime_df = filter_by_era(df, ["early_war", "late_war"])
    """
    if isinstance(era_values, str):
        era_values = [era_values]

    return df.filter(pl.col(era_column).is_in(era_values))
