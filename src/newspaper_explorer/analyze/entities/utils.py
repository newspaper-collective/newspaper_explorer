"""
Compare entity extraction methods.

This module provides utilities for comparing entity extraction results
of different methods (GLiNER, LLM, etc.) using the QueryEngine.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import polars as pl

from newspaper_explorer.config.base import get_config
from newspaper_explorer.analyze.query.engine import QueryEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def list_entity_results(source_name: str) -> List[Dict[str, Any]]:
    """
    List all available entity extraction results for a source.

    Args:
        source_name: Name of the newspaper source

    Returns:
        List of dicts with method_id, method_type, model, timestamp, entity_count
    """
    config = get_config()
    entities_dir = Path(config.results_dir) / source_name / "entities"

    if not entities_dir.exists():
        return []

    results = []
    for method_dir in sorted(entities_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        metadata_path = method_dir / "metadata.json"
        entities_path = method_dir / "entities.parquet"

        if not metadata_path.exists() or not entities_path.exists():
            continue

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Count entities
        df = pl.read_parquet(entities_path)
        entity_count = len(df)

        results.append(
            {
                "method_id": method_dir.name,
                "method_type": metadata.get("method_type", "unknown"),
                "model": metadata.get("parameters", {}).get("model", "unknown"),
                "timestamp": metadata.get("timestamp", "unknown"),
                "entity_count": entity_count,
                "line_count": metadata.get("line_count", 0),
                "duration": metadata.get("duration_seconds", 0),
            }
        )

    return results


def load_result_metadata(source_name: str, method_id: str) -> Dict[str, Any]:
    """
    Load metadata for a specific entity extraction result.

    Args:
        source_name: Name of the newspaper source
        method_id: Method ID to load

    Returns:
        Metadata dictionary
    """
    config = get_config()
    metadata_path = (
        Path(config.results_dir) / source_name / "entities" / method_id / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))

def compare_existing_results(
    source_name: str, method_id_1: str, method_id_2: str
) -> Dict[str, Any]:
    """
    Compare two existing entity extraction results without re-running.

    Args:
        source_name: Name of the newspaper source
        method_id_1: First method ID to compare
        method_id_2: Second method ID to compare

    Returns:
        Dict with metadata and comparison results
    """
    # Load metadata for both methods
    metadata_1 = load_result_metadata(source_name, method_id_1)
    metadata_2 = load_result_metadata(source_name, method_id_2)

    # Create results dict matching run_both_methods format
    config = get_config()
    results = {
        "method_1": {
            "metadata": metadata_1,
            "output_dir": Path(config.results_dir) / source_name / "entities" / method_id_1,
        },
        "method_2": {
            "metadata": metadata_2,
            "output_dir": Path(config.results_dir) / source_name / "entities" / method_id_2,
        },
    }

    # Run comparison
    comparisons = compare_entities(source_name, method_id_1, method_id_2)

    return {"results": results, "comparisons": comparisons}


def compare_coverage(source_name: str, method_id_1: str, method_id_2: str) -> pl.DataFrame:
    """
    Compare which lines each method extracted entities from.

    Args:
        source_name: Name of the newspaper source
        method_id_1: First method ID
        method_id_2: Second method ID

    Returns:
        DataFrame comparing coverage
    """
    config = get_config()
    path_1 = Path(config.results_dir) / source_name / "entities" / method_id_1 / "entities.parquet"
    path_2 = Path(config.results_dir) / source_name / "entities" / method_id_2 / "entities.parquet"

    with QueryEngine(source=source_name) as qe:
        comparison = qe.query(
            f"""
            WITH method1_lines AS (
                SELECT DISTINCT source_id 
                FROM '{path_1}'
            ),
            method2_lines AS (
                SELECT DISTINCT source_id
                FROM '{path_2}'
            ),
            source_lines AS (
                SELECT DISTINCT source_id
                FROM lines
            )
            SELECT
                COUNT(DISTINCT s.source_id) as total_lines,
                COUNT(DISTINCT m1.source_id) as method1_coverage,
                COUNT(DISTINCT m2.source_id) as method2_coverage,
                COUNT(DISTINCT CASE WHEN m1.source_id IS NOT NULL AND m2.source_id IS NOT NULL 
                      THEN s.source_id END) as both_methods,
                COUNT(DISTINCT CASE WHEN m1.source_id IS NOT NULL AND m2.source_id IS NULL 
                      THEN s.source_id END) as method1_only,
                COUNT(DISTINCT CASE WHEN m1.source_id IS NULL AND m2.source_id IS NOT NULL 
                      THEN s.source_id END) as method2_only
            FROM source_lines s
            LEFT JOIN method1_lines m1 ON s.source_id = m1.source_id
            LEFT JOIN method2_lines m2 ON s.source_id = m2.source_id
        """
        )

    return cast(pl.DataFrame, comparison)


def compare_entities(
    source_name: str, method_id_1: str, method_id_2: str
) -> Dict[str, pl.DataFrame]:
    """
    Compare entities found by two methods.

    Args:
        source_name: Name of the newspaper source
        method_id_1: First method ID
        method_id_2: Second method ID

    Returns:
        Dict with comparison DataFrames
    """
    config = get_config()
    path_1 = Path(config.results_dir) / source_name / "entities" / method_id_1 / "entities.parquet"
    path_2 = Path(config.results_dir) / source_name / "entities" / method_id_2 / "entities.parquet"

    with QueryEngine(source=source_name) as qe:
        # Overall statistics
        stats = qe.query(
            f"""
            SELECT
                'Method 1' as method,
                COUNT(*) as total_entities,
                COUNT(DISTINCT entity_text) as unique_entities,
                COUNT(DISTINCT source_id) as lines_with_entities
            FROM '{path_1}'
            UNION ALL
            SELECT
                'Method 2' as method,
                COUNT(*) as total_entities,
                COUNT(DISTINCT entity_text) as unique_entities,
                COUNT(DISTINCT source_id) as lines_with_entities
            FROM '{path_2}'
        """
        )

        # Entity type distribution
        type_dist = qe.query(
            f"""
            WITH combined AS (
                SELECT 'Method 1' as method, entity_type, source_id
                FROM '{path_1}'
                UNION ALL
                SELECT 'Method 2' as method, entity_type, source_id
                FROM '{path_2}'
            )
            SELECT
                method,
                entity_type,
                COUNT(*) as count,
                COUNT(DISTINCT source_id) as unique_lines
            FROM combined
            GROUP BY method, entity_type
            ORDER BY method, count DESC
        """
        )

        # Entities found by both methods (normalized comparison)
        both = qe.query(
            f"""
            WITH method1_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text as method1_text
                FROM '{path_1}'
            ),
            method2_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text as method2_text
                FROM '{path_2}'
            )
            SELECT
                m1.method1_text,
                m2.method2_text,
                m1.entity_type,
                COUNT(*) as agreement_count
            FROM method1_entities m1
            INNER JOIN method2_entities m2
                ON m1.entity_lower = m2.entity_lower 
                AND m1.entity_type = m2.entity_type
            GROUP BY m1.method1_text, m2.method2_text, m1.entity_type
            ORDER BY agreement_count DESC
            LIMIT 50
        """
        )

        # Entities unique to each method
        method1_only = qe.query(
            f"""
            WITH method1_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text
                FROM '{path_1}'
            ),
            method2_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type
                FROM '{path_2}'
            )
            SELECT
                m1.entity_text,
                m1.entity_type,
                COUNT(*) as mentions
            FROM method1_entities m1
            LEFT JOIN method2_entities m2
                ON m1.entity_lower = m2.entity_lower 
                AND m1.entity_type = m2.entity_type
            WHERE m2.entity_lower IS NULL
            GROUP BY m1.entity_text, m1.entity_type
            ORDER BY mentions DESC
            LIMIT 25
        """
        )

        method2_only = qe.query(
            f"""
            WITH method1_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type
                FROM '{path_1}'
            ),
            method2_entities AS (
                SELECT DISTINCT 
                    LOWER(TRIM(entity_text)) as entity_lower,
                    entity_type,
                    entity_text
                FROM '{path_2}'
            )
            SELECT
                m2.entity_text,
                m2.entity_type,
                COUNT(*) as mentions
            FROM method2_entities m2
            LEFT JOIN method1_entities m1
                ON m2.entity_lower = m1.entity_lower 
                AND m2.entity_type = m1.entity_type
            WHERE m1.entity_lower IS NULL
            GROUP BY m2.entity_text, m2.entity_type
            ORDER BY mentions DESC
            LIMIT 25
        """
        )

    return {
        "statistics": stats,
        "type_distribution": type_dist,
        "agreement": both,
        "method1_only": method1_only,
        "method2_only": method2_only,
    }


def print_comparison_report(comparison_data: Dict[str, Any]):
    """
    Print a formatted comparison report for any two methods.

    Args:
        comparison_data: Output from compare_existing_results()
    """
    results = comparison_data["results"]
    comparisons = comparison_data["comparisons"]

    method_1 = results["method_1"]["metadata"]
    method_2 = results["method_2"]["metadata"]

    print("\n" + "=" * 80)
    print("ENTITY EXTRACTION METHOD COMPARISON")
    print("=" * 80)

    # Execution metadata
    print("\n### Method 1 ###\n")
    print(f"Method ID:    {method_1['analysis_id']}")
    print(f"Method Type:  {method_1.get('method_type', 'unknown')}")
    print(f"Model:        {method_1.get('parameters', {}).get('model', 'unknown')}")
    print(f"Duration:     {method_1.get('duration_seconds', 0):.1f}s")
    print(f"Line Count:   {method_1.get('line_count', 0)}")

    print("\n### Method 2 ###\n")
    print(f"Method ID:    {method_2['analysis_id']}")
    print(f"Method Type:  {method_2.get('method_type', 'unknown')}")
    print(f"Model:        {method_2.get('parameters', {}).get('model', 'unknown')}")
    print(f"Duration:     {method_2.get('duration_seconds', 0):.1f}s")
    print(f"Line Count:   {method_2.get('line_count', 0)}")

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
    print("\n### Entities Unique to Method 1 ###\n")
    print(comparisons["method1_only"].head(10))

    print("\n### Entities Unique to Method 2 ###\n")
    print(comparisons["method2_only"].head(10))

    print("\n" + "=" * 80)
