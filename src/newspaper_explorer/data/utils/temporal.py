"""
Temporal metadata utilities for historical newspaper analysis.

Provides era classification based on year/date columns with configurable
presets for common historical periodizations (WWI, Weimar, etc.).

Usage:
    from newspaper_explorer.data.utils.temporal import add_era_columns
    from newspaper_explorer.models.data.temporal import ERA_PRESETS, get_preset

    # Add era columns using a preset
    df = add_era_columns(df, preset="wwi")

    # Or with custom era definition
    df = add_era_columns(df, eras={
        "early": (1900, 1910),
        "late": (1911, 1920),
    })

    # Or with EraPreset model directly
    preset = get_preset("wwi")
    df = add_era_columns(df, era_preset=preset)
"""

import logging
from typing import Optional, Union

import polars as pl

from newspaper_explorer.models.data.temporal import (
    ERA_PRESETS,
    EraPreset,
    get_preset,
    list_presets,
)

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "ERA_PRESETS",
    "EraPreset",
    "add_era_columns",
    "add_multiple_era_columns",
    "filter_by_era",
    "get_era_counts",
    "get_preset",
    "list_presets",
]

# Type alias for legacy era definitions: era_name -> (start_year, end_year)
EraDefinitionDict = dict[str, tuple[Optional[int], Optional[int]]]


def _extract_year_from_column(
    df: pl.DataFrame,
    year_column: str,
    date_column: str,
) -> pl.Expr:
    """
    Create a Polars expression to extract year from available columns.

    Tries year_column first, then extracts from date_column if needed.
    """
    # Check which columns exist
    has_year = year_column in df.columns
    has_date = date_column in df.columns

    if has_year:
        return pl.col(year_column).cast(pl.Int64)

    if has_date:
        # Handle both datetime and date types
        col = pl.col(date_column)
        dtype = df.schema.get(date_column)
        if isinstance(dtype, (pl.Date, pl.Datetime)):
            return col.dt.year()
        # Try parsing as string
        return col.str.to_datetime().dt.year()

    msg = f"Neither year column '{year_column}' nor date column '{date_column}' found in DataFrame"
    raise ValueError(msg)


def _resolve_era_definitions(
    preset: Optional[str],
    era_preset: Optional[EraPreset],
    eras: Optional[EraDefinitionDict],
) -> EraDefinitionDict:
    """
    Resolve era definitions from the provided source.

    Validates that exactly one source is provided and returns the era dict.
    """
    # Validate inputs - exactly one source must be provided
    sources_provided = sum([preset is not None, era_preset is not None, eras is not None])
    if sources_provided == 0:
        msg = "Must provide one of: 'preset', 'era_preset', or 'eras' argument"
        raise ValueError(msg)
    if sources_provided > 1:
        msg = "Provide only one of: 'preset', 'era_preset', or 'eras' argument"
        raise ValueError(msg)

    # Get era definitions as dict
    if preset is not None:
        preset_model = get_preset(preset)
        logger.info(f"Using era preset '{preset}': {preset_model.name}")
        return preset_model.get_era_dict()

    if era_preset is not None:
        logger.info(f"Using era preset '{era_preset.key}': {era_preset.name}")
        return era_preset.get_era_dict()

    # eras is not None here due to validation above
    logger.info(f"Using custom eras: {list(eras.keys())}")  # type: ignore[union-attr]
    return eras  # type: ignore[return-value]


def _build_era_condition(
    year_expr: pl.Expr,
    start: Optional[int],
    end: Optional[int],
) -> pl.Expr:
    """Build a Polars condition expression for an era's year range."""
    conditions: list[pl.Expr] = []
    if start is not None:
        conditions.append(year_expr >= start)
    if end is not None:
        conditions.append(year_expr <= end)

    if not conditions:
        return pl.lit(value=True)
    if len(conditions) == 1:
        return conditions[0]
    return conditions[0] & conditions[1]


def add_era_columns(
    df: pl.DataFrame,
    *,
    preset: Optional[str] = None,
    era_preset: Optional[EraPreset] = None,
    eras: Optional[EraDefinitionDict] = None,
    year_column: str = "year",
    date_column: str = "date",
    output_column: str = "era",
) -> pl.DataFrame:
    """
    Add era classification column(s) to a DataFrame based on year/date.

    Can use a preset name, EraPreset model, or custom era definition.
    At least one of `preset`, `era_preset`, or `eras` must be provided.

    Args:
        df: Input DataFrame with year or date column
        preset: Name of era preset to use (e.g., "wwi", "war_phases")
        era_preset: EraPreset model instance to use directly
        eras: Custom era definition as dict mapping era names to (start, end) tuples.
              Use None for open-ended ranges: {"pre": (None, 1913), "post": (1914, None)}
        year_column: Name of year column (default: "year")
        date_column: Name of date/datetime column to extract year from (default: "date")
        output_column: Name for the output era column (default: "era")

    Returns:
        DataFrame with added era column.

    Raises:
        ValueError: If no era definition provided, or if required columns missing.

    Example:
        >>> from newspaper_explorer.data.utils.temporal import add_era_columns
        >>> from newspaper_explorer.models.data.temporal import get_preset
        >>>
        >>> # Using a preset name
        >>> df = add_era_columns(df, preset="wwi")
        >>>
        >>> # Using an EraPreset model
        >>> preset = get_preset("war_phases")
        >>> df = add_era_columns(df, era_preset=preset)
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
    # Resolve era definitions from provided source
    era_defs = _resolve_era_definitions(preset, era_preset, eras)

    # Extract year expression
    year_expr = _extract_year_from_column(df, year_column, date_column)

    # Build when/then/otherwise chain for era classification
    era_items = list(era_defs.items())
    first_name, (first_start, first_end) = era_items[0]

    # Build the expression chain using type: ignore for Polars chained types
    chain = pl.when(_build_era_condition(year_expr, first_start, first_end)).then(
        pl.lit(first_name)
    )

    for era_name, (start, end) in era_items[1:]:
        chain = chain.when(_build_era_condition(year_expr, start, end)).then(  # type: ignore[assignment]
            pl.lit(era_name)
        )

    era_expr = chain.otherwise(pl.lit("")).alias(output_column)

    return df.with_columns(era_expr)


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
