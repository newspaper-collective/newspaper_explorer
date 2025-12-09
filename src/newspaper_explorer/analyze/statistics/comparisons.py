"""
High-level comparison functions for temporal analysis.

Combines hypothesis tests, effect sizes, and multiple testing corrections
into convenient functions for comparing groups across eras.

Usage:
    from newspaper_explorer.analyze.statistics.comparisons import compare_eras

    # Compare emotions across WWI eras
    results = compare_eras(
        df,
        value_cols=["Joy_prob", "Fear_prob", "Anger_prob"],
        era_col="era",
    )
"""

from itertools import combinations
import logging
from typing import Optional, Union

import polars as pl

from newspaper_explorer.analyze.statistics.corrections import correct_pvalues
from newspaper_explorer.analyze.statistics.effects import cliffs_delta, cohens_d
from newspaper_explorer.analyze.statistics.tests import kruskal_wallis, mann_whitney_u

logger = logging.getLogger(__name__)


def compare_two_groups(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: tuple[str, str],
    *,
    test: str = "mann_whitney_u",
    effect_size: str = "cliffs_delta",
) -> dict:
    """
    Compare two groups with hypothesis test and effect size.

    Args:
        df: Input DataFrame
        value_col: Column containing values to compare
        group_col: Column containing group labels
        groups: Tuple of (group1, group2) to compare
        test: Test to use ('mann_whitney_u' or 'ttest')
        effect_size: Effect size measure ('cliffs_delta' or 'cohens_d')

    Returns:
        Dictionary with test results, effect size, and group statistics.
    """
    group1, group2 = groups

    # Get values
    vals1 = df.filter(pl.col(group_col) == group1).select(value_col).to_series().drop_nulls()
    vals2 = df.filter(pl.col(group_col) == group2).select(value_col).to_series().drop_nulls()

    if len(vals1) == 0 or len(vals2) == 0:
        logger.warning(
            f"Empty group(s) for {value_col}: {group1}={len(vals1)}, {group2}={len(vals2)}"
        )
        return {
            "value_col": value_col,
            "group1": group1,
            "group2": group2,
            "n1": len(vals1),
            "n2": len(vals2),
            "p_value": None,
            "effect_size": None,
            "error": "Empty group(s)",
        }

    # Run test
    test_result = mann_whitney_u(df, value_col, group_col, groups)

    # Calculate effect size
    if effect_size == "cliffs_delta":
        es_value = cliffs_delta(vals1, vals2)
    else:
        es_value = cohens_d(vals1, vals2)

    return {
        "value_col": value_col,
        "group1": group1,
        "group2": group2,
        "n1": len(vals1),
        "n2": len(vals2),
        "mean1": float(vals1.mean()),
        "mean2": float(vals2.mean()),
        "median1": float(vals1.median()),
        "median2": float(vals2.median()),
        "statistic": test_result["statistic"],
        "p_value": test_result["p_value"],
        "effect_size": es_value,
        "effect_size_method": effect_size,
        "test": test,
    }


def compare_multiple_groups(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[list[str]] = None,
    *,
    test: str = "kruskal_wallis",
) -> dict:
    """
    Compare multiple groups with omnibus test.

    Args:
        df: Input DataFrame
        value_col: Column containing values to compare
        group_col: Column containing group labels
        groups: List of groups to include (None = all)
        test: Test to use ('kruskal_wallis' or 'anova')

    Returns:
        Dictionary with omnibus test results and group statistics.
    """
    result = kruskal_wallis(df, value_col, group_col, groups)
    result["value_col"] = value_col
    return result


def pairwise_comparisons(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[list[str]] = None,
    *,
    correction: str = "benjamini_hochberg",
    alpha: float = 0.05,
) -> pl.DataFrame:
    """
    Perform all pairwise comparisons between groups with correction.

    Args:
        df: Input DataFrame
        value_col: Column containing values to compare
        group_col: Column containing group labels
        groups: List of groups to compare (None = all)
        correction: P-value correction method
        alpha: Significance threshold

    Returns:
        DataFrame with pairwise comparison results including corrected p-values.
    """
    # Get groups
    if groups is None:
        groups = df.select(pl.col(group_col).unique()).to_series().sort().to_list()
        groups = [g for g in groups if g]  # Remove empty strings

    if len(groups) < 2:
        logger.warning(f"Need at least 2 groups, found {len(groups)}")
        return pl.DataFrame()

    # Run all pairwise comparisons
    results = []
    p_values = {}

    for g1, g2 in combinations(groups, 2):
        result = compare_two_groups(df, value_col, group_col, (g1, g2))
        pair_key = f"{g1}_vs_{g2}"
        if result["p_value"] is not None:
            p_values[pair_key] = result["p_value"]
        results.append(result)

    # Apply multiple testing correction
    if p_values:
        corrected = correct_pvalues(p_values, method=correction, alpha=alpha)

        # Add corrected values to results
        for result in results:
            pair_key = f"{result['group1']}_vs_{result['group2']}"
            if pair_key in corrected:
                corr = corrected[pair_key]
                result["p_adjusted"] = corr.get("q", corr.get("p_adjusted"))
                result["significant"] = corr["significant"]

    return pl.DataFrame(results)


def compare_eras(
    df: pl.DataFrame,
    value_cols: Union[str, list[str]],
    era_col: str = "era",
    *,
    eras: Optional[list[str]] = None,
    pairwise: bool = True,
    correction: str = "benjamini_hochberg",
    alpha: float = 0.05,
    effect_size: str = "cliffs_delta",
) -> dict:
    """
    Compare values across historical eras with full statistical analysis.

    Performs omnibus test (Kruskal-Wallis) plus pairwise comparisons
    with multiple testing correction.

    Args:
        df: Input DataFrame with era column
        value_cols: Column(s) to compare (e.g., ["Joy_prob", "Fear_prob"])
        era_col: Column containing era labels
        eras: List of eras to include (None = all non-empty)
        pairwise: If True, include pairwise comparisons
        correction: P-value correction method for pairwise tests
        alpha: Significance threshold
        effect_size: Effect size method for pairwise comparisons

    Returns:
        Dictionary with:
            - omnibus: Dict of omnibus test results per value column
            - pairwise: DataFrame of pairwise comparisons (if pairwise=True)
            - summary: Summary DataFrame with key statistics

    Example:
        >>> from newspaper_explorer.data.utils.temporal import add_era_columns
        >>> from newspaper_explorer.analyze.statistics.comparisons import compare_eras
        >>>
        >>> # Add era classification
        >>> df = add_era_columns(df, preset="wwi")
        >>>
        >>> # Compare emotions across eras
        >>> results = compare_eras(
        ...     df,
        ...     value_cols=["Joy_prob", "Fear_prob", "Anger_prob"],
        ...     era_col="era",
        ... )
        >>>
        >>> # Check omnibus results
        >>> for col, res in results["omnibus"].items():
        ...     print(f"{col}: H={res['statistic']:.2f}, p={res['p_value']:.4f}")
        >>>
        >>> # View pairwise comparisons
        >>> print(results["pairwise"])
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    # Get eras
    if eras is None:
        eras = df.select(pl.col(era_col).unique()).to_series().sort().to_list()
        eras = [e for e in eras if e]

    logger.info(f"Comparing {len(value_cols)} value columns across {len(eras)} eras: {eras}")

    # Omnibus tests
    omnibus_results = {}
    for col in value_cols:
        omnibus_results[col] = compare_multiple_groups(df, col, era_col, eras)

    # Pairwise comparisons
    pairwise_results = None
    if pairwise and len(eras) >= 2:
        all_pairwise = []
        for col in value_cols:
            pw = pairwise_comparisons(
                df,
                col,
                era_col,
                eras,
                correction=correction,
                alpha=alpha,
            )
            all_pairwise.append(pw)

        if all_pairwise:
            pairwise_results = pl.concat(all_pairwise)

    # Build summary
    summary_rows = []
    for col, omni in omnibus_results.items():
        row = {
            "variable": col,
            "n_groups": omni["n_groups"],
            "H_statistic": omni["statistic"],
            "p_value": omni["p_value"],
            "significant": omni["p_value"] < alpha,
        }

        # Add group means
        for group, stats in omni.get("group_stats", {}).items():
            row[f"mean_{group}"] = stats["mean"]
            row[f"n_{group}"] = stats["n"]

        summary_rows.append(row)

    summary = pl.DataFrame(summary_rows)

    return {
        "omnibus": omnibus_results,
        "pairwise": pairwise_results,
        "summary": summary,
        "eras": eras,
        "alpha": alpha,
        "correction": correction,
    }


def find_best_era_comparison(
    df: pl.DataFrame,
    value_col: str,
    era_col: str = "era",
    eras: Optional[list[str]] = None,
    *,
    correction: str = "benjamini_hochberg",
    alpha: float = 0.05,
) -> dict:
    """
    Find the most significant era pair comparison for a value column.

    Useful when you want to identify which era transition shows the
    strongest effect.

    Args:
        df: Input DataFrame
        value_col: Column to compare
        era_col: Era column name
        eras: List of eras to consider
        correction: P-value correction method
        alpha: Significance threshold

    Returns:
        Dictionary with best comparison result, or None if no significant result.
    """
    pairwise = pairwise_comparisons(
        df,
        value_col,
        era_col,
        eras,
        correction=correction,
        alpha=alpha,
    )

    if pairwise.is_empty():
        return None

    # Find row with smallest adjusted p-value
    best = pairwise.sort("p_adjusted").head(1).to_dicts()[0]

    return {
        "value_col": value_col,
        "group1": best["group1"],
        "group2": best["group2"],
        "p_value": best["p_value"],
        "p_adjusted": best["p_adjusted"],
        "effect_size": best["effect_size"],
        "significant": best.get("significant", False),
        "mean_diff": best["mean1"] - best["mean2"],
    }
