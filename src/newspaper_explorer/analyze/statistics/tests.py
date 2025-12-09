"""
Statistical hypothesis tests for group comparisons.

Provides non-parametric and parametric tests using scipy.stats:
- Mann-Whitney U test (non-parametric, 2 groups)
- Kruskal-Wallis H test (non-parametric, 3+ groups)
- Independent t-test (parametric, 2 groups)
- One-way ANOVA (parametric, 3+ groups)

All functions work with Polars DataFrames and return structured results.

Usage:
    from newspaper_explorer.analyze.statistics.tests import mann_whitney_u, kruskal_wallis

    # Compare two groups
    result = mann_whitney_u(df, value_col="Joy", group_col="era", groups=["pre", "post"])

    # Compare multiple groups
    result = kruskal_wallis(df, value_col="Joy", group_col="era")
"""

import logging
from typing import Optional

import polars as pl
from scipy import stats

logger = logging.getLogger(__name__)


def mann_whitney_u(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[tuple[str, str]] = None,
    alternative: str = "two-sided",
) -> dict:
    """
    Perform Mann-Whitney U test comparing two groups.

    Non-parametric test for comparing two independent samples. Does not assume
    normal distribution - ideal for emotion scores, counts, etc.

    Args:
        df: Input DataFrame
        value_col: Column containing values to compare
        group_col: Column containing group labels
        groups: Tuple of (group1, group2) to compare. If None, uses first two unique values.
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test results:
            - statistic: U statistic
            - p_value: p-value
            - group1, group2: Group names
            - n1, n2: Sample sizes
            - median1, median2: Group medians
            - mean1, mean2: Group means

    Example:
        >>> result = mann_whitney_u(df, "Joy_prob", "era", groups=("pre_war", "war"))
        >>> print(f"p = {result['p_value']:.4f}")
    """
    # Get group labels
    if groups is None:
        unique_groups = df.select(pl.col(group_col).unique()).to_series().to_list()
        if len(unique_groups) < 2:
            msg = f"Need at least 2 groups, found {len(unique_groups)}"
            raise ValueError(msg)
        groups = (unique_groups[0], unique_groups[1])
        logger.info(f"Auto-selected groups: {groups}")

    group1, group2 = groups

    # Extract values for each group
    vals1 = (
        df.filter(pl.col(group_col) == group1).select(value_col).to_series().drop_nulls().to_list()
    )
    vals2 = (
        df.filter(pl.col(group_col) == group2).select(value_col).to_series().drop_nulls().to_list()
    )

    if len(vals1) == 0 or len(vals2) == 0:
        msg = f"One or both groups have no data: {group1}={len(vals1)}, {group2}={len(vals2)}"
        raise ValueError(msg)

    # Perform test
    stat_result = stats.mannwhitneyu(vals1, vals2, alternative=alternative)

    return {
        "test": "mann_whitney_u",
        "statistic": float(stat_result.statistic),
        "p_value": float(stat_result.pvalue),
        "group1": group1,
        "group2": group2,
        "n1": len(vals1),
        "n2": len(vals2),
        "median1": float(pl.Series(vals1).median()),
        "median2": float(pl.Series(vals2).median()),
        "mean1": float(pl.Series(vals1).mean()),
        "mean2": float(pl.Series(vals2).mean()),
        "alternative": alternative,
    }


def kruskal_wallis(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[list[str]] = None,
) -> dict:
    """
    Perform Kruskal-Wallis H test comparing multiple groups.

    Non-parametric equivalent of one-way ANOVA. Tests whether samples
    from different groups originate from the same distribution.

    Args:
        df: Input DataFrame
        value_col: Column containing values to compare
        group_col: Column containing group labels
        groups: List of groups to include. If None, uses all unique values.

    Returns:
        Dictionary with test results:
            - statistic: H statistic
            - p_value: p-value
            - groups: List of group names
            - n_groups: Number of groups
            - group_stats: Dict with n, median, mean per group

    Example:
        >>> result = kruskal_wallis(df, "Fear_prob", "era")
        >>> if result['p_value'] < 0.05:
        ...     print("Significant difference between eras")
    """
    # Get groups
    if groups is None:
        groups = df.select(pl.col(group_col).unique()).to_series().sort().to_list()
        # Filter out empty strings
        groups = [g for g in groups if g]

    if len(groups) < 2:
        msg = f"Need at least 2 groups, found {len(groups)}"
        raise ValueError(msg)

    # Extract values for each group
    group_values = []
    group_stats = {}

    for group in groups:
        vals = (
            df.filter(pl.col(group_col) == group)
            .select(value_col)
            .to_series()
            .drop_nulls()
            .to_list()
        )
        if len(vals) == 0:
            logger.warning(f"Group '{group}' has no data, skipping")
            continue
        group_values.append(vals)
        group_stats[group] = {
            "n": len(vals),
            "median": float(pl.Series(vals).median()),
            "mean": float(pl.Series(vals).mean()),
        }

    if len(group_values) < 2:
        msg = "Need at least 2 non-empty groups for Kruskal-Wallis test"
        raise ValueError(msg)

    # Perform test
    stat_result = stats.kruskal(*group_values)

    return {
        "test": "kruskal_wallis",
        "statistic": float(stat_result.statistic),
        "p_value": float(stat_result.pvalue),
        "groups": list(group_stats.keys()),
        "n_groups": len(group_stats),
        "group_stats": group_stats,
    }


def independent_ttest(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[tuple[str, str]] = None,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> dict:
    """
    Perform independent samples t-test comparing two groups.

    Parametric test assuming approximately normal distributions.
    Use Welch's t-test (equal_var=False) when variances may differ.

    Args:
        df: Input DataFrame
        value_col: Column containing values to compare
        group_col: Column containing group labels
        groups: Tuple of (group1, group2) to compare
        equal_var: If True, assume equal variances (Student's t-test).
                   If False, use Welch's t-test (recommended default).
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test results including statistic, p_value, group stats.
    """
    if groups is None:
        unique_groups = df.select(pl.col(group_col).unique()).to_series().to_list()
        if len(unique_groups) < 2:
            msg = f"Need at least 2 groups, found {len(unique_groups)}"
            raise ValueError(msg)
        groups = (unique_groups[0], unique_groups[1])

    group1, group2 = groups

    vals1 = (
        df.filter(pl.col(group_col) == group1).select(value_col).to_series().drop_nulls().to_list()
    )
    vals2 = (
        df.filter(pl.col(group_col) == group2).select(value_col).to_series().drop_nulls().to_list()
    )

    if len(vals1) == 0 or len(vals2) == 0:
        msg = f"One or both groups have no data: {group1}={len(vals1)}, {group2}={len(vals2)}"
        raise ValueError(msg)

    stat_result = stats.ttest_ind(vals1, vals2, equal_var=equal_var, alternative=alternative)

    return {
        "test": "welch_ttest" if not equal_var else "student_ttest",
        "statistic": float(stat_result.statistic),
        "p_value": float(stat_result.pvalue),
        "group1": group1,
        "group2": group2,
        "n1": len(vals1),
        "n2": len(vals2),
        "mean1": float(pl.Series(vals1).mean()),
        "mean2": float(pl.Series(vals2).mean()),
        "std1": float(pl.Series(vals1).std()),
        "std2": float(pl.Series(vals2).std()),
        "equal_var": equal_var,
        "alternative": alternative,
    }


def one_way_anova(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[list[str]] = None,
) -> dict:
    """
    Perform one-way ANOVA comparing multiple groups.

    Parametric test assuming normal distributions and equal variances.
    For non-normal data, prefer kruskal_wallis().

    Args:
        df: Input DataFrame
        value_col: Column containing values to compare
        group_col: Column containing group labels
        groups: List of groups to include

    Returns:
        Dictionary with test results including F statistic, p_value, group stats.
    """
    if groups is None:
        groups = df.select(pl.col(group_col).unique()).to_series().sort().to_list()
        groups = [g for g in groups if g]

    if len(groups) < 2:
        msg = f"Need at least 2 groups, found {len(groups)}"
        raise ValueError(msg)

    group_values = []
    group_stats = {}

    for group in groups:
        vals = (
            df.filter(pl.col(group_col) == group)
            .select(value_col)
            .to_series()
            .drop_nulls()
            .to_list()
        )
        if len(vals) == 0:
            continue
        group_values.append(vals)
        group_stats[group] = {
            "n": len(vals),
            "mean": float(pl.Series(vals).mean()),
            "std": float(pl.Series(vals).std()),
        }

    if len(group_values) < 2:
        msg = "Need at least 2 non-empty groups for ANOVA"
        raise ValueError(msg)

    stat_result = stats.f_oneway(*group_values)

    return {
        "test": "one_way_anova",
        "statistic": float(stat_result.statistic),
        "p_value": float(stat_result.pvalue),
        "groups": list(group_stats.keys()),
        "n_groups": len(group_stats),
        "group_stats": group_stats,
    }
