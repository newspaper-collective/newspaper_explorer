"""
Example usage of GLiNER-based entity extraction.

Demonstrates the updated extraction pipeline following the data architecture pattern.
"""

import logging

from newspaper_explorer.analysis.entities.gliner_extraction import (
    GLiNEREntityExtractor,
    extract_entities_gliner,
)
from newspaper_explorer.analysis.query.engine import QueryEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def example_simple_extraction():
    """Simple extraction using convenience function."""
    print("=== Simple GLiNER Extraction (First 50 Lines) ===\n")

    # Extract from first 50 lines (testing)
    results = extract_entities_gliner(source_name="der_tag", threshold=0.5, batch_size=16, limit=50)

    print(f"\nResults saved to: {results['output_dir']}")
    print(f"Method ID: {results['metadata']['analysis_id']}")
    print(f"Total entities: {len(results['results_df'])}")
    print(f"Duration: {results['metadata']['duration_seconds']:.1f}s")


def example_custom_extraction():
    """Extraction with custom configuration."""
    print("\n=== Custom GLiNER Extraction Configuration ===\n")

    # Initialize extractor with custom settings
    extractor = GLiNEREntityExtractor(
        source_name="der_tag",
        model_name="urchade/gliner_multi-v2.1",
        threshold=0.6,  # Higher threshold for precision
        batch_size=64,  # Larger batch if GPU available
        min_text_length=50,  # Lower minimum
        normalize=True,
    )

    # Run extraction on first 100 lines
    results = extractor.extract_and_save(limit=100)

    # Show results
    print(f"\nExtracted {len(results['results_df'])} entities")
    print("\nSample entities:")
    print(results["results_df"].head(10))

    # Entity type distribution
    type_counts = results["results_df"]["entity_type"].value_counts()
    print("\nEntity type distribution:")
    print(type_counts)


def example_query_results():
    """Query extracted entities using QueryEngine."""
    print("\n=== Querying GLiNER Extracted Entities ===\n")

    # First, run a small extraction
    results = extract_entities_gliner(source_name="der_tag", limit=100)
    method_id = results["metadata"]["analysis_id"]

    # Now query the results
    with QueryEngine(source="der_tag") as qe:
        # Find all person entities
        persons = qe.query(
            f"""
            SELECT 
                e.entity_text,
                COUNT(*) as mention_count,
                AVG(e.confidence) as avg_confidence
            FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
            WHERE e.entity_type = 'person'
            GROUP BY e.entity_text
            ORDER BY mention_count DESC
            LIMIT 10
        """
        )

        print("Top persons mentioned:")
        print(persons)

        # Get high-confidence locations
        locations = qe.query(
            f"""
            SELECT 
                e.entity_text,
                COUNT(*) as mentions,
                MAX(e.confidence) as max_confidence
            FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
            WHERE e.entity_type = 'location' AND e.confidence > 0.7
            GROUP BY e.entity_text
            ORDER BY mentions DESC
            LIMIT 10
        """
        )

        print("\nHigh-confidence locations:")
        print(locations)


def example_compare_with_llm():
    """Compare GLiNER with LLM entity extraction."""
    print("\n=== Comparing GLiNER vs LLM ===\n")

    # Note: This assumes you have LLM results
    # Run GLiNER extraction
    gliner_results = extract_entities_gliner(source_name="der_tag", limit=50)
    gliner_method = gliner_results["metadata"]["analysis_id"]

    print(f"GLiNER method: {gliner_method}")
    print(f"GLiNER entities: {len(gliner_results['results_df'])}")
    print(f"GLiNER duration: {gliner_results['metadata']['duration_seconds']:.1f}s")

    # If you have LLM results, compare:
    # with QueryEngine(source="der_tag") as qe:
    #     comparison = qe.compare_entity_methods(
    #         method1=gliner_method,
    #         method2="llm_gpt4o_mini_20241019"
    #     )
    #     print(f"\nDifferences between methods: {len(comparison)}")


def example_entity_statistics():
    """Get entity extraction statistics."""
    print("\n=== Entity Extraction Statistics ===\n")

    results = extract_entities_gliner(source_name="der_tag", limit=200)
    df = results["results_df"]

    print(f"Total entities: {len(df)}")
    print(f"Unique entities: {df['entity_text'].n_unique()}")
    print(f"\nBy type:")

    for entity_type in df["entity_type"].unique():
        type_df = df.filter(pl.col("entity_type") == entity_type)
        print(f"  {entity_type}: {len(type_df)} ({type_df['entity_text'].n_unique()} unique)")

    print(f"\nConfidence statistics:")
    print(f"  Mean: {df['confidence'].mean():.3f}")
    print(f"  Min: {df['confidence'].min():.3f}")
    print(f"  Max: {df['confidence'].max():.3f}")


if __name__ == "__main__":
    import polars as pl

    print("GLiNER Entity Extraction Examples")
    print("=" * 60)

    # Example 1: Simple extraction
    example_simple_extraction()

    # Example 2: Custom configuration
    # example_custom_extraction()

    # Example 3: Query results
    # example_query_results()

    # Example 4: Statistics
    # example_entity_statistics()

    # Example 5: Compare with LLM
    # example_compare_with_llm()
