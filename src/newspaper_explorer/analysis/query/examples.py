"""
Example usage of query engine for efficient data access.

Demonstrates DuckDB-based queries for exploring analysis results
without loading multi-GB Parquet files into memory.
"""

from newspaper_explorer.analysis.query.engine import QueryEngine


def example_search_text():
    """Search for text content across all lines."""
    with QueryEngine(source="der_tag") as qe:
        # Search for mentions of "Berlin"
        results = qe.search_text(
            query_text="Berlin", start_date="1901-01-01", end_date="1901-12-31", limit=10
        )

        print(f"Found {len(results)} mentions:")
        for row in results.iter_rows(named=True):
            print(f"[{row['date']}] {row['text'][:100]}...")


def example_entity_mentions():
    """Find all mentions of a specific entity."""
    with QueryEngine(source="der_tag") as qe:
        # Find all mentions of Kaiser Wilhelm II
        mentions = qe.find_entity_mentions(
            entity_name="Kaiser Wilhelm II", method="llm_gpt4o_mini", start_date="1901-01-01"
        )

        print(f"Found {len(mentions)} mentions of Kaiser Wilhelm II:")
        for row in mentions.head(10).iter_rows(named=True):
            print(f"[{row['date']}] {row['text'][:100]}...")


def example_entity_frequency():
    """Get entity mention frequency over time."""
    with QueryEngine(source="der_tag") as qe:
        # Get top entities by year
        freq = qe.entity_frequency(
            method="llm_gpt4o_mini", entity_type="person", min_mentions=10, group_by="year"
        )

        print("Top entities by year:")
        print(freq.head(20))


def example_compare_methods():
    """Compare results from different extraction methods."""
    with QueryEngine(source="der_tag") as qe:
        # Compare LLM vs spaCy entity extraction
        comparison = qe.compare_entity_methods(
            method1="llm_gpt4o_mini", method2="spacy_de_core_news_lg"
        )

        print(f"Found {len(comparison)} differences between methods:")
        print(comparison.head(10))


def example_topic_distribution():
    """Get topic distribution over time."""
    with QueryEngine(source="der_tag") as qe:
        # Get topics by year
        topics = qe.get_topic_distribution(method="llm_gpt4o_mini", group_by="year")

        print("Topic distribution by year:")
        print(topics.head(20))


def example_custom_query():
    """Execute custom SQL query."""
    with QueryEngine(source="der_tag") as qe:
        # Custom query: find co-occurrences of entities
        result = qe.query(
            """
            SELECT 
                e1.entity_text as entity1,
                e2.entity_text as entity2,
                COUNT(*) as co_occurrence_count
            FROM 'results/der_tag/entities/llm_gpt4o_mini/entities.parquet' e1
            JOIN 'results/der_tag/entities/llm_gpt4o_mini/entities.parquet' e2
                ON e1.line_id = e2.line_id
                AND e1.entity_text < e2.entity_text
            GROUP BY entity1, entity2
            HAVING co_occurrence_count > 5
            ORDER BY co_occurrence_count DESC
            LIMIT 20
        """
        )

        print("Entity co-occurrences:")
        print(result)


def example_get_line_context():
    """Get full context for a specific line."""
    with QueryEngine(source="der_tag") as qe:
        # Get line details
        line = qe.get_line(line_id="der_tag_1901_01_15_001_block_003_line_001")

        if line:
            print(f"Line ID: {line['line_id']}")
            print(f"Date: {line['date']}")
            print(f"Text: {line['text']}")
            print(f"Filename: {line['filename']}")
        else:
            print("Line not found")


if __name__ == "__main__":
    print("=== Text Search ===")
    example_search_text()

    print("\n=== Entity Mentions ===")
    example_entity_mentions()

    print("\n=== Entity Frequency ===")
    example_entity_frequency()

    print("\n=== Line Context ===")
    example_get_line_context()
