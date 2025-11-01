"""
Compare LLM and GLiNER entity extraction methods.

This module demonstrates how to run both methods and compare their results
using the QueryEngine for side-by-side analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import polars as pl

from newspaper_explorer.analysis.entities.llm_extraction import extract_entities_llm
from newspaper_explorer.analysis.entities.gliner_extraction import extract_entities_gliner
from newspaper_explorer.config.base import get_config
from newspaper_explorer.analysis.query.engine import QueryEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_both_methods(source_name: str = "der_tag", limit: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    Run both LLM and GLiNER extraction on the same data.

    Args:
        source_name: Name of the newspaper source
        limit: Number of lines to process (for testing)

    Returns:
        Dict with 'llm' and 'gliner' keys containing results
    """
    logger.info(f"Running both extraction methods on {limit} lines from {source_name}")

    # Run LLM extraction
    logger.info("Starting LLM extraction...")
    llm_results = extract_entities_llm(source_name=source_name, limit=limit)

    # Run GLiNER extraction
    logger.info("Starting GLiNER extraction...")
    gliner_results = extract_entities_gliner(source_name=source_name, limit=limit)

    return {"llm": llm_results, "gliner": gliner_results}


def compare_coverage(source_name: str, llm_method_id: str, gliner_method_id: str) -> pl.DataFrame:
    """
    Compare which lines each method extracted entities from.

    Args:
        source_name: Name of the newspaper source
        llm_method_id: Method ID for LLM results
        gliner_method_id: Method ID for GLiNER results

    Returns:
        DataFrame comparing coverage
    """
    config = get_config()
    llm_path = (
        Path(config.results_dir) / source_name / "entities" / llm_method_id / "entities.parquet"
    )
    gliner_path = (
        Path(config.results_dir) / source_name / "entities" / gliner_method_id / "entities.parquet"
    )

    with QueryEngine(source=source_name) as qe:
        comparison = qe.query(
            f"""
            WITH llm_lines AS (
                SELECT DISTINCT line_id 
                FROM '{llm_path}'
            ),
            gliner_lines AS (
                SELECT DISTINCT line_id
                FROM '{gliner_path}'
            ),
            source_lines AS (
                SELECT DISTINCT line_id
                FROM lines
            )
            SELECT
                COUNT(DISTINCT s.line_id) as total_lines,
                COUNT(DISTINCT l.line_id) as llm_coverage,
                COUNT(DISTINCT g.line_id) as gliner_coverage,
                COUNT(DISTINCT CASE WHEN l.line_id IS NOT NULL AND g.line_id IS NOT NULL 
                      THEN s.line_id END) as both_methods,
                COUNT(DISTINCT CASE WHEN l.line_id IS NOT NULL AND g.line_id IS NULL 
                      THEN s.line_id END) as llm_only,
                COUNT(DISTINCT CASE WHEN l.line_id IS NULL AND g.line_id IS NOT NULL 
                      THEN s.line_id END) as gliner_only
            FROM source_lines s
            LEFT JOIN llm_lines l ON s.line_id = l.line_id
            LEFT JOIN gliner_lines g ON s.line_id = g.line_id
        """
        )

    return comparison


def compare_entities(
    source_name: str, llm_method_id: str, gliner_method_id: str
) -> Dict[str, pl.DataFrame]:
    """
    Compare entities found by each method.

    Args:
        source_name: Name of the newspaper source
        llm_method_id: Method ID for LLM results
        gliner_method_id: Method ID for GLiNER results

    Returns:
        Dict with comparison DataFrames
    """
    config = get_config()
    llm_path = (
        Path(config.results_dir) / source_name / "entities" / llm_method_id / "entities.parquet"
    )
    gliner_path = (
        Path(config.results_dir) / source_name / "entities" / gliner_method_id / "entities.parquet"
    )

    with QueryEngine(source=source_name) as qe:
        # Overall statistics
        stats = qe.query(
            f"""
            SELECT
                'LLM' as method,
                COUNT(*) as total_entities,
                COUNT(DISTINCT entity_text) as unique_entities,
                COUNT(DISTINCT line_id) as lines_with_entities
            FROM '{llm_path}'
            UNION ALL
            SELECT
                'GLiNER' as method,
                COUNT(*) as total_entities,
                COUNT(DISTINCT entity_text) as unique_entities,
                COUNT(DISTINCT line_id) as lines_with_entities
            FROM '{gliner_path}'
        """
        )

        # Entity type distribution
        types = qe.query(
            f"""
            SELECT
                'LLM' as method,
                entity_type,
                COUNT(*) as count
            FROM '{llm_path}'
            GROUP BY entity_type
            UNION ALL
            SELECT
                'GLiNER' as method,
                entity_type,
                COUNT(*) as count
            FROM '{gliner_path}'
            GROUP BY entity_type
            ORDER BY method, entity_type
        """
        )

        # Entities found by both methods (normalized comparison)
        both = qe.query(
            f"""
            WITH llm_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text as llm_text
                FROM '{llm_path}'
            ),
            gliner_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text as gliner_text
                FROM '{gliner_path}'
            )
            SELECT
                l.llm_text,
                g.gliner_text,
                l.entity_type,
                COUNT(*) as agreement_count
            FROM llm_entities l
            INNER JOIN gliner_entities g 
                ON l.entity_lower = g.entity_lower 
                AND l.entity_type = g.entity_type
            GROUP BY l.llm_text, g.gliner_text, l.entity_type
            ORDER BY agreement_count DESC
            LIMIT 50
        """
        )

        # Entities unique to each method
        llm_only = qe.query(
            f"""
            WITH llm_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text
                FROM '{llm_path}'
            ),
            gliner_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type
                FROM '{gliner_path}'
            )
            SELECT
                l.entity_text,
                l.entity_type,
                COUNT(*) as llm_mentions
            FROM llm_entities l
            LEFT JOIN gliner_entities g 
                ON l.entity_lower = g.entity_lower 
                AND l.entity_type = g.entity_type
            WHERE g.entity_lower IS NULL
            GROUP BY l.entity_text, l.entity_type
            ORDER BY llm_mentions DESC
            LIMIT 25
        """
        )

        gliner_only = qe.query(
            f"""
            WITH llm_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type
                FROM '{llm_path}'
            ),
            gliner_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text
                FROM '{gliner_path}'
            )
            SELECT
                g.entity_text,
                g.entity_type,
                COUNT(*) as gliner_mentions
            FROM gliner_entities g
            LEFT JOIN llm_entities l 
                ON g.entity_lower = l.entity_lower 
                AND g.entity_type = l.entity_type
            WHERE l.entity_lower IS NULL
            GROUP BY g.entity_text, g.entity_type
            ORDER BY gliner_mentions DESC
            LIMIT 25
        """
        )

    return {
        "statistics": stats,
        "type_distribution": types,
        "agreement": both,
        "llm_only": llm_only,
        "gliner_only": gliner_only,
    }


def print_comparison_report(
    results: Dict[str, Dict[str, Any]], comparisons: Dict[str, pl.DataFrame]
):
    """
    Print a formatted comparison report.

    Args:
        results: Results from run_both_methods()
        comparisons: Comparison DataFrames from compare_entities()
    """
    llm = results["llm"]
    gliner = results["gliner"]

    print("\n" + "=" * 80)
    print("ENTITY EXTRACTION METHOD COMPARISON")
    print("=" * 80)

    # Execution metadata
    print("\n### Execution Metadata ###\n")
    print(f"LLM Method ID:    {llm['metadata']['analysis_id']}")
    print(f"LLM Model:        {llm['metadata']['parameters']['model_name']}")
    print(f"LLM Duration:     {llm['metadata']['duration_seconds']:.1f}s")
    print(f"LLM Temperature:  {llm['metadata']['parameters']['temperature']}")
    print()
    print(f"GLiNER Method ID: {gliner['metadata']['analysis_id']}")
    print(f"GLiNER Model:     {gliner['metadata']['parameters']['model_name']}")
    print(f"GLiNER Duration:  {gliner['metadata']['duration_seconds']:.1f}s")
    print(f"GLiNER Threshold: {gliner['metadata']['parameters']['threshold']}")

    # Overall statistics
    print("\n### Overall Statistics ###\n")
    print(comparisons["statistics"])

    # Entity type distribution
    print("\n### Entity Type Distribution ###\n")
    print(comparisons["type_distribution"])

    # Agreement
    print("\n### Top Entities Found by Both Methods ###\n")
    print(comparisons["agreement"].head(10))

    # Method-specific entities
    print("\n### Entities Found ONLY by LLM ###\n")
    print(comparisons["llm_only"].head(10))

    print("\n### Entities Found ONLY by GLiNER ###\n")
    print(comparisons["gliner_only"].head(10))

    print("\n" + "=" * 80)


def main():
    """Run complete comparison pipeline."""
    # Configuration
    SOURCE = "der_tag"
    LIMIT = 50  # Small sample for testing

    print("Running entity extraction comparison...")
    print(f"Source: {SOURCE}")
    print(f"Sample size: {LIMIT} lines\n")

    # Step 1: Run both methods
    results = run_both_methods(source_name=SOURCE, limit=LIMIT)

    # Step 2: Compare results
    comparisons = compare_entities(
        source_name=SOURCE,
        llm_method_id=results["llm"]["metadata"]["analysis_id"],
        gliner_method_id=results["gliner"]["metadata"]["analysis_id"],
    )

    # Step 3: Print report
    print_comparison_report(results, comparisons)

    # Step 4: Coverage comparison
    coverage = compare_coverage(
        source_name=SOURCE,
        llm_method_id=results["llm"]["metadata"]["analysis_id"],
        gliner_method_id=results["gliner"]["metadata"]["analysis_id"],
    )

    print("\n### Line Coverage Comparison ###\n")
    print(coverage)

    print(
        f"\nResults saved to:\nLLM:    {results['llm']['output_dir']}\nGLiNER: {results['gliner']['output_dir']}"
    )


if __name__ == "__main__":
    main()
