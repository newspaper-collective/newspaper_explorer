"""
Multiple testing corrections for p-values.

Provides correction methods for controlling false discovery rate (FDR)
or family-wise error rate (FWER) when performing multiple comparisons:
- Benjamini-Hochberg (FDR control, recommended for exploratory analysis)
- Benjamini-Yekutieli (FDR control, more conservative)
- Bonferroni (FWER control, very conservative)
- Holm-Bonferroni (FWER control, less conservative than Bonferroni)

Usage:
    from newspaper_explorer.analyze.statistics.corrections import benjamini_hochberg

    # Correct p-values from multiple tests
    corrected = benjamini_hochberg(p_values)
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)


def benjamini_hochberg(
    p_values: Union[list[float], dict[str, float]],
    alpha: float = 0.05,
) -> Union[list[dict], dict[str, dict]]:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.

    Controls the False Discovery Rate (FDR) - the expected proportion of
    false positives among rejected hypotheses. Less conservative than
    Bonferroni, appropriate for exploratory analysis.

    Args:
        p_values: List of p-values or dict mapping labels to p-values
        alpha: Significance threshold (default: 0.05)

    Returns:
        If input is list: List of dicts with 'p', 'q' (adjusted p), 'significant', 'rank'
        If input is dict: Dict mapping labels to result dicts

    Example:
        >>> p_vals = {"Joy": 0.01, "Fear": 0.03, "Anger": 0.08, "Sadness": 0.02}
        >>> results = benjamini_hochberg(p_vals)
        >>> for emotion, r in results.items():
        ...     sig = "***" if r['significant'] else ""
        ...     print(f"{emotion}: p={r['p']:.3f}, q={r['q']:.3f} {sig}")
    """
    # Handle dict input
    is_dict = isinstance(p_values, dict)
    if is_dict:
        labels = list(p_values.keys())
        p_list = [p_values[k] for k in labels]
    else:
        labels = None
        p_list = list(p_values)

    m = len(p_list)
    if m == 0:
        return {} if is_dict else []

    # Sort p-values and track original indices
    indexed = sorted(enumerate(p_list), key=lambda x: x[1])

    # Calculate adjusted p-values (q-values)
    q_values = [0.0] * m

    for i, (orig_idx, p) in enumerate(indexed):
        rank = i + 1
        # BH adjustment: q = p * m / rank
        q = p * m / rank
        q_values[orig_idx] = q

    # Enforce monotonicity (ensure q increases with rank)
    # Work backwards from largest to smallest
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(m - 2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        q_values[idx] = min(q_values[idx], q_values[next_idx])

    # Cap at 1.0
    q_values = [min(1.0, q) for q in q_values]

    # Build results
    results = []
    for i, (p, q) in enumerate(zip(p_list, q_values)):
        results.append(
            {
                "p": p,
                "q": q,
                "significant": q < alpha,
                "rank": sorted([pv for pv in p_list]).index(p) + 1,
            }
        )

    if is_dict:
        return {label: results[i] for i, label in enumerate(labels)}
    return results


def benjamini_yekutieli(
    p_values: Union[list[float], dict[str, float]],
    alpha: float = 0.05,
) -> Union[list[dict], dict[str, dict]]:
    """
    Apply Benjamini-Yekutieli FDR correction.

    More conservative than Benjamini-Hochberg. Controls FDR under
    arbitrary dependency structure between tests.

    Args:
        p_values: List of p-values or dict mapping labels to p-values
        alpha: Significance threshold

    Returns:
        Same format as benjamini_hochberg()
    """
    is_dict = isinstance(p_values, dict)
    if is_dict:
        labels = list(p_values.keys())
        p_list = [p_values[k] for k in labels]
    else:
        labels = None
        p_list = list(p_values)

    m = len(p_list)
    if m == 0:
        return {} if is_dict else []

    # Correction factor: c(m) = sum(1/i) for i in 1..m
    c_m = sum(1 / i for i in range(1, m + 1))

    # Sort and calculate q-values
    indexed = sorted(enumerate(p_list), key=lambda x: x[1])
    q_values = [0.0] * m

    for i, (orig_idx, p) in enumerate(indexed):
        rank = i + 1
        # BY adjustment: q = p * m * c(m) / rank
        q = p * m * c_m / rank
        q_values[orig_idx] = q

    # Enforce monotonicity
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(m - 2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        q_values[idx] = min(q_values[idx], q_values[next_idx])

    q_values = [min(1.0, q) for q in q_values]

    results = []
    for i, (p, q) in enumerate(zip(p_list, q_values)):
        results.append(
            {
                "p": p,
                "q": q,
                "significant": q < alpha,
            }
        )

    if is_dict:
        return {label: results[i] for i, label in enumerate(labels)}
    return results


def bonferroni(
    p_values: Union[list[float], dict[str, float]],
    alpha: float = 0.05,
) -> Union[list[dict], dict[str, dict]]:
    """
    Apply Bonferroni correction for multiple testing.

    Controls Family-Wise Error Rate (FWER) - probability of making
    any false positive. Very conservative - appropriate when false
    positives have high cost.

    Args:
        p_values: List of p-values or dict mapping labels to p-values
        alpha: Significance threshold

    Returns:
        Same format as benjamini_hochberg()
    """
    is_dict = isinstance(p_values, dict)
    if is_dict:
        labels = list(p_values.keys())
        p_list = [p_values[k] for k in labels]
    else:
        labels = None
        p_list = list(p_values)

    m = len(p_list)
    if m == 0:
        return {} if is_dict else []

    # Bonferroni: adjusted p = p * m
    adjusted_alpha = alpha / m

    results = []
    for p in p_list:
        p_adj = min(1.0, p * m)
        results.append(
            {
                "p": p,
                "p_adjusted": p_adj,
                "significant": p < adjusted_alpha,
                "adjusted_alpha": adjusted_alpha,
            }
        )

    if is_dict:
        return {label: results[i] for i, label in enumerate(labels)}
    return results


def holm_bonferroni(
    p_values: Union[list[float], dict[str, float]],
    alpha: float = 0.05,
) -> Union[list[dict], dict[str, dict]]:
    """
    Apply Holm-Bonferroni (step-down) correction.

    Controls FWER like Bonferroni but is uniformly more powerful.
    Tests are ordered by p-value and compared to progressively
    less strict thresholds.

    Args:
        p_values: List of p-values or dict mapping labels to p-values
        alpha: Significance threshold

    Returns:
        Same format as benjamini_hochberg()
    """
    is_dict = isinstance(p_values, dict)
    if is_dict:
        labels = list(p_values.keys())
        p_list = [p_values[k] for k in labels]
    else:
        labels = None
        p_list = list(p_values)

    m = len(p_list)
    if m == 0:
        return {} if is_dict else []

    # Sort by p-value
    indexed = sorted(enumerate(p_list), key=lambda x: x[1])

    # Step through and determine significance
    significant = [False] * m
    p_adjusted = [0.0] * m

    reject = True
    for i, (orig_idx, p) in enumerate(indexed):
        rank = i + 1
        threshold = alpha / (m - rank + 1)

        # Holm step-down: once we fail to reject, stop
        if reject and p <= threshold:
            significant[orig_idx] = True
        else:
            reject = False

        # Adjusted p-value
        p_adj = min(1.0, p * (m - rank + 1))
        p_adjusted[orig_idx] = p_adj

    # Enforce monotonicity on adjusted p-values
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(1, m):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        p_adjusted[idx] = max(p_adjusted[idx], p_adjusted[prev_idx])

    results = []
    for i, p in enumerate(p_list):
        results.append(
            {
                "p": p,
                "p_adjusted": p_adjusted[i],
                "significant": significant[i],
            }
        )

    if is_dict:
        return {label: results[i] for i, label in enumerate(labels)}
    return results


def correct_pvalues(
    p_values: Union[list[float], dict[str, float]],
    method: str = "benjamini_hochberg",
    alpha: float = 0.05,
) -> Union[list[dict], dict[str, dict]]:
    """
    Apply multiple testing correction using specified method.

    Convenience function for selecting correction method.

    Args:
        p_values: List or dict of p-values
        method: Correction method:
            - 'benjamini_hochberg' or 'bh' or 'fdr': BH FDR control (default)
            - 'benjamini_yekutieli' or 'by': BY FDR control
            - 'bonferroni': Bonferroni FWER control
            - 'holm' or 'holm_bonferroni': Holm-Bonferroni FWER control
        alpha: Significance threshold

    Returns:
        Corrected results (format depends on input type)

    Example:
        >>> p_vals = {"Joy": 0.01, "Fear": 0.03, "Anger": 0.08}
        >>> results = correct_pvalues(p_vals, method="bh")
    """
    method = method.lower()

    if method in ("benjamini_hochberg", "bh", "fdr"):
        return benjamini_hochberg(p_values, alpha)
    elif method in ("benjamini_yekutieli", "by"):
        return benjamini_yekutieli(p_values, alpha)
    elif method == "bonferroni":
        return bonferroni(p_values, alpha)
    elif method in ("holm", "holm_bonferroni"):
        return holm_bonferroni(p_values, alpha)
    else:
        msg = f"Unknown method: {method}. Use 'bh', 'by', 'bonferroni', or 'holm'"
        raise ValueError(msg)
