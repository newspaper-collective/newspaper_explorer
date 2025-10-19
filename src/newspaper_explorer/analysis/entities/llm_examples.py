"""
Example usage of LLM-based entity extraction.

Demonstrates the new extraction pipeline following the data architecture pattern.
"""

import logging

from newspaper_explorer.analysis.entities.llm_extraction import (
    LLMEntityExtractor,
    extract_entities_llm,
)
from newspaper_explorer.utils.queries import QueryEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def example_simple_extraction():
    """Simple extraction using convenience function."""
    print("=== Simple Extraction (First 10 Lines) ===\n")

    # Extract from first 10 lines (testing)
    results = extract_entities_llm(
        source_name="der_tag", model_name="gpt-4o-mini", temperature=0.3, limit=10
    )

    print(f"\nResults saved to: {results['output_dir']}")
    print(f"Method ID: {results['metadata']['analysis_id']}")
    print(f"Total entities: {len(results['results_df'])}")


def example_custom_extraction():
    """Extraction with custom configuration."""
    print("\n=== Custom Extraction Configuration ===\n")

    # Initialize extractor with custom settings
    extractor = LLMEntityExtractor(
        source_name="der_tag",
        model_name="gpt-4o-mini",
        temperature=0.2,  # More deterministic
        max_tokens=1500,
        max_retries=5,  # More retries for reliability
    )

    # Run extraction on first 20 lines
    results = extractor.extract_and_save(limit=20)

    # Show results
    print(f"\nExtracted {len(results['results_df'])} entities")
    print("\nSample entities:")
    print(results["results_df"].head(10))


def example_query_results():
    """Query extracted entities using QueryEngine."""
    print("\n=== Querying Extracted Entities ===\n")

    # First, run a small extraction
    results = extract_entities_llm(source_name="der_tag", limit=50)
    method_id = results["metadata"]["analysis_id"]

    # Now query the results
    with QueryEngine(source="der_tag") as qe:
        # Find all person entities
        persons = qe.query(
            f"""
            SELECT 
                e.entity_text,
                COUNT(*) as mention_count
            FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
            WHERE e.entity_type = 'person'
            GROUP BY e.entity_text
            ORDER BY mention_count DESC
            LIMIT 10
        """
        )

        print("Top persons mentioned:")
        print(persons)

        # Find entity with context
        if len(persons) > 0:
            top_person = persons["entity_text"][0]
            mentions = qe.find_entity_mentions(entity_name=top_person, method=method_id)

            print(f"\nMentions of '{top_person}':")
            for row in mentions.head(3).iter_rows(named=True):
                print(f"[{row['date']}] {row['text'][:100]}...")


def example_compare_methods():
    """Compare LLM vs GLiNER entity extraction."""
    print("\n=== Method Comparison (LLM vs Traditional) ===\n")

    # Note: This assumes you have both LLM and GLiNER results
    # Run LLM extraction
    llm_results = extract_entities_llm(source_name="der_tag", limit=30)
    llm_method = llm_results["metadata"]["analysis_id"]

    print(f"LLM method: {llm_method}")
    print(f"LLM entities: {len(llm_results['results_df'])}")

    # If you have GLiNER results, you can compare:
    # with QueryEngine(source="der_tag") as qe:
    #     comparison = qe.compare_entity_methods(
    #         method1=llm_method,
    #         method2="gliner_multi_v2_1_20241019"
    #     )
    #     print(f"\nDifferences: {len(comparison)}")


if __name__ == "__main__":
    # Run examples (requires .env with LLM_BASE_URL and LLM_API_KEY)

    print("Entity Extraction Examples")
    print("=" * 60)

    # Example 1: Simple extraction
    example_simple_extraction()

    # Example 2: Custom configuration
    # example_custom_extraction()

    # Example 3: Query results
    # example_query_results()

    # Example 4: Compare methods
    # example_compare_methods()
