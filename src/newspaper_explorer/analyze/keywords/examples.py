"""
Examples demonstrating TF-IDF keyword extraction.

Shows various use cases for extracting keywords from newspaper texts.
"""

from pathlib import Path
from newspaper_explorer.analyze.keywords.tf_idf import (
    TFIDFExtractor,
    extract_keywords_from_document,
)


def example_basic_extraction():
    """Basic keyword extraction from text blocks."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Keyword Extraction")
    print("=" * 80 + "\n")

    extractor = TFIDFExtractor(
        source_name="der_tag",
        use_stopwords=True,
    )

    # Extract keywords for each text block
    results = extractor.extract_keywords(
        top_k=10,
        limit=100,  # Process first 100 blocks for demo
    )

    print(f"Extracted keywords for {len(results):,} text blocks")
    print("\nSample results:")
    print(results.head(3))

    # Save results
    output_file = extractor.save_results(results, output_name="textblock_keywords")
    print(f"\nSaved to: {output_file}")


def example_grouped_by_date():
    """Extract keywords grouped by publication date."""
    print("\n" + "=" * 80)
    print("Example 2: Keywords Grouped by Date")
    print("=" * 80 + "\n")

    extractor = TFIDFExtractor(source_name="der_tag")

    # Group by date - get daily keywords
    results = extractor.extract_keywords(
        group_by="date",
        top_k=15,
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.8,  # Ignore words in more than 80% of documents
        limit=1000,
    )

    print(f"Extracted keywords for {len(results):,} dates")
    print("\nSample daily keywords:")
    for row in results.head(3).iter_rows(named=True):
        print(f"\nDate: {row['date']}")
        print(f"Top keywords: {', '.join(row['keywords'][:10])}")


def example_grouped_by_year():
    """Extract keywords grouped by year."""
    print("\n" + "=" * 80)
    print("Example 3: Keywords Grouped by Year")
    print("=" * 80 + "\n")

    extractor = TFIDFExtractor(source_name="der_tag")

    # Group by year - see how topics evolved over time
    results = extractor.extract_keywords(
        group_by="year",
        top_k=20,
        ngram_range=(1, 2),  # Include bigrams for better context
    )

    print(f"Extracted keywords for {len(results):,} years")
    print("\nYearly keywords:")
    for row in results.iter_rows(named=True):
        print(f"\n{row['year']}:")
        keywords = row["keywords"][:10]
        scores = row["scores"][:10]
        for kw, score in zip(keywords, scores):
            print(f"  {kw:30s} {score:.4f}")


def example_with_bigrams():
    """Extract keywords including bigrams (2-word phrases)."""
    print("\n" + "=" * 80)
    print("Example 4: Keywords with Bigrams")
    print("=" * 80 + "\n")

    extractor = TFIDFExtractor(source_name="der_tag")

    # Include bigrams to capture phrases like "erste weltkrieg"
    results = extractor.extract_keywords(
        top_k=15,
        ngram_range=(1, 2),  # Unigrams + bigrams
        limit=100,
    )

    print("Sample keywords with phrases:")
    for row in results.head(3).iter_rows(named=True):
        print(f"\nDocument ID: {row.get('doc_id', 'N/A')}")
        bigrams = [kw for kw in row["keywords"] if " " in kw]
        if bigrams:
            print(f"Bigrams found: {', '.join(bigrams[:5])}")


def example_with_preprocessed_text():
    """Use preprocessed/normalized text for better results."""
    print("\n" + "=" * 80)
    print("Example 5: Using Preprocessed Text")
    print("=" * 80 + "\n")

    # Check if normalized text exists
    normalized_file = Path("data/processed/der_tag/textblocks_normalized.parquet")

    if not normalized_file.exists():
        print(f"Normalized file not found: {normalized_file}")
        print("Run preprocessing first:")
        print("  newspaper-explorer data preprocess --source der_tag")
        return

    extractor = TFIDFExtractor(
        source_name="der_tag",
        input_file=normalized_file,
        text_column="text_normalized",
    )

    results = extractor.extract_keywords(
        group_by="date",
        top_k=10,
        limit=500,
    )

    print(f"Extracted keywords from {len(results):,} normalized documents")


def example_single_document():
    """Extract keywords from a single document given a corpus."""
    print("\n" + "=" * 80)
    print("Example 6: Single Document Keyword Extraction")
    print("=" * 80 + "\n")

    # Example corpus
    corpus = [
        "der krieg hat viele opfer gefordert",
        "die wirtschaft wächst stetig",
        "neue gesetze wurden verabschiedet",
        "der kaiser spricht zum volk",
        "wissenschaftler machen fortschritte",
    ]

    # New document
    document = "der krieg endete mit vielen opfern und großen verlusten"

    # Extract keywords
    result = extract_keywords_from_document(
        document=document,
        corpus=corpus,
        top_k=5,
    )

    print("Document:", document)
    print("\nTop keywords:")
    for kw, score in zip(result["keywords"], result["scores"]):
        print(f"  {kw:20s} {score:.4f}")


def example_custom_stopwords():
    """Use custom stopwords to filter domain-specific common words."""
    print("\n" + "=" * 80)
    print("Example 7: Custom Stopwords")
    print("=" * 80 + "\n")

    # Add newspaper-specific stopwords
    custom_stopwords = [
        "zeitung",
        "seite",
        "ausgabe",
        "nummer",
        "tag",
        "heute",
        "gestern",
        "morgen",
    ]

    extractor = TFIDFExtractor(
        source_name="der_tag",
        use_stopwords=True,  # Use default German stopwords
        custom_stopwords=custom_stopwords,  # Plus custom ones
    )

    results = extractor.extract_keywords(
        group_by="date",
        top_k=10,
        limit=100,
    )

    print(f"Extracted keywords with {len(custom_stopwords)} custom stopwords")
    print("Custom stopwords filtered:", ", ".join(custom_stopwords))


def example_different_thresholds():
    """Compare results with different min_df and max_df thresholds."""
    print("\n" + "=" * 80)
    print("Example 8: Comparing Thresholds")
    print("=" * 80 + "\n")

    extractor = TFIDFExtractor(source_name="der_tag")

    # Strict filtering - only keep words that appear in at least 5 docs
    # but not more than 50% of docs
    print("Strict filtering (min_df=5, max_df=0.5):")
    results_strict = extractor.extract_keywords(
        group_by="year",
        top_k=10,
        min_df=5,
        max_df=0.5,
    )
    print(f"  Found keywords for {len(results_strict):,} years")

    # Loose filtering - keep rare words too
    print("\nLoose filtering (min_df=1, max_df=0.9):")
    results_loose = extractor.extract_keywords(
        group_by="year",
        top_k=10,
        min_df=1,
        max_df=0.9,
    )
    print(f"  Found keywords for {len(results_loose):,} years")


if __name__ == "__main__":
    """
    Run examples. Comment out examples you don't want to run.

    Prerequisites:
        newspaper-explorer data parse --source der_tag
    """

    # Basic examples
    example_basic_extraction()
    example_grouped_by_date()
    example_grouped_by_year()

    # Advanced examples
    example_with_bigrams()
    example_single_document()
    example_custom_stopwords()
    example_different_thresholds()

    # Requires preprocessing
    # example_with_preprocessed_text()

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("=" * 80 + "\n")
