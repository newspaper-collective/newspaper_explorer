"""
Examples demonstrating TF-IDF keyword extraction.

Shows various use cases for extracting keywords from newspaper texts.
"""

from pathlib import Path
from newspaper_explorer.analyze.keywords.tf_idf import (
    TFIDFExtractor,
    extract_keywords_from_document,
)
from newspaper_explorer.analyze.keywords.lda import (
    LDAExtractor,
    extract_topic_keywords_simple,
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


def example_lda_train_model():
    """Train an LDA topic model on newspaper texts."""
    print("\n" + "=" * 80)
    print("Example 9: LDA - Train Topic Model")
    print("=" * 80 + "\n")

    extractor = LDAExtractor(source_name="der_tag")

    # Train model with 15 topics
    print("Training LDA model with 15 topics...")
    model_info = extractor.train_model(
        num_topics=15,
        passes=10,
        limit=5000,  # Use first 5000 documents for demo
    )

    print(f"\nModel trained!")
    print(f"Status: {model_info['status']}")
    print(f"Documents: {model_info.get('num_documents', 'N/A'):,}")
    print(f"Vocabulary: {model_info.get('vocabulary_size', 'N/A'):,} tokens")
    print(f"Perplexity: {model_info.get('perplexity', 'N/A'):.4f}")
    print(f"Saved to: {model_info['model_path']}")


def example_lda_topic_keywords():
    """Extract keywords for each discovered topic."""
    print("\n" + "=" * 80)
    print("Example 10: LDA - Topic Keywords")
    print("=" * 80 + "\n")

    # Use the simple one-shot function
    print("Training model and extracting topic keywords...")
    topics_df = extract_topic_keywords_simple(
        source_name="der_tag",
        num_topics=10,
        top_k=15,
        limit=2000,  # Small corpus for demo
    )

    print(f"\nExtracted keywords for {len(topics_df)} topics\n")

    # Show all topics
    for row in topics_df.iter_rows(named=True):
        topic_id = row["topic_id"]
        keywords = row["keywords"][:10]
        scores = row["scores"][:10]

        print(f"Topic {topic_id}:")
        kw_str = ", ".join([f"{kw}({s:.3f})" for kw, s in zip(keywords, scores)])
        print(f"  {kw_str}\n")


def example_lda_document_keywords():
    """Assign topic-based keywords to documents."""
    print("\n" + "=" * 80)
    print("Example 11: LDA - Document Keywords")
    print("=" * 80 + "\n")

    extractor = LDAExtractor(source_name="der_tag")

    # Train model first (or load existing)
    print("Training model...")
    extractor.train_model(num_topics=10, passes=10, limit=2000)

    # Extract document keywords
    print("\nExtracting document keywords...")
    doc_keywords = extractor.extract_document_keywords(
        top_k=5,
        min_topic_prob=0.1,
        limit=100,
    )

    print(f"Extracted keywords for {len(doc_keywords):,} documents")
    print("\nSample results:")
    for row in doc_keywords.head(3).iter_rows(named=True):
        doc_id = row["doc_id"]
        keywords = row["keywords"]
        topics = row.get("topics", [])
        topic_probs = row.get("topic_probs", [])

        print(f"\nDocument: {doc_id}")
        if topics:
            topic_info = ", ".join([f"T{t}({p:.2f})" for t, p in zip(topics[:3], topic_probs[:3])])
            print(f"  Main topics: {topic_info}")
        print(f"  Keywords: {', '.join(keywords)}")


def example_lda_yearly_topics():
    """Discover topics in yearly aggregated data."""
    print("\n" + "=" * 80)
    print("Example 12: LDA - Yearly Topic Evolution")
    print("=" * 80 + "\n")

    extractor = LDAExtractor(source_name="der_tag")

    # Train on yearly aggregated data
    print("Training model on yearly aggregated texts...")
    model_info = extractor.train_model(
        num_topics=10,
        passes=15,
        group_by=["year"],  # Aggregate all text per year
    )

    print(f"\nModel trained on {model_info.get('num_documents', 'N/A')} years")

    # Extract topic keywords
    topics_df = extractor.get_topic_keywords(top_k=10)

    print("\nTopic keywords (representing yearly themes):")
    for row in topics_df.head(5).iter_rows(named=True):
        topic_id = row["topic_id"]
        keywords = row["keywords"][:8]
        print(f"Topic {topic_id}: {', '.join(keywords)}")


def example_lda_vs_tfidf():
    """Compare LDA and TF-IDF keyword extraction."""
    print("\n" + "=" * 80)
    print("Example 13: LDA vs TF-IDF Comparison")
    print("=" * 80 + "\n")

    # TF-IDF - document-specific keywords
    print("TF-IDF (document-specific keywords):")
    tfidf_extractor = TFIDFExtractor(source_name="der_tag")
    tfidf_results = tfidf_extractor.extract_keywords(
        document_level="textblock",
        top_k=5,
        limit=10,
    )
    print(f"  Extracted keywords for {len(tfidf_results):,} documents")
    sample = tfidf_results.head(2)
    for row in sample.iter_rows(named=True):
        print(f"  Doc keywords: {', '.join(row['keywords'])}")

    print("\nLDA (topic-based keywords):")
    lda_extractor = LDAExtractor(source_name="der_tag")
    lda_extractor.train_model(num_topics=5, passes=5, limit=100)
    lda_results = lda_extractor.extract_document_keywords(top_k=5, limit=10)
    print(f"  Extracted keywords for {len(lda_results):,} documents")
    sample = lda_results.head(2)
    for row in sample.iter_rows(named=True):
        topics = row.get("topics", [])
        print(f"  Doc keywords (from topics {topics[:2]}): {', '.join(row['keywords'])}")

    print("\nKey difference:")
    print("  - TF-IDF: Unique words specific to each document")
    print("  - LDA: Thematic words from corpus-wide topics")


if __name__ == "__main__":
    """
    Run examples. Comment out examples you don't want to run.

    Prerequisites:
        newspaper-explorer data parse --source der_tag

    Note: LDA examples require gensim:
        pip install -e '.[nlp]'
    """

    # TF-IDF examples
    print("=" * 80)
    print("TF-IDF EXAMPLES")
    print("=" * 80)

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

    # LDA examples
    print("\n\n" + "=" * 80)
    print("LDA EXAMPLES")
    print("=" * 80)

    # Uncomment to run LDA examples (requires gensim)
    # example_lda_train_model()
    # example_lda_topic_keywords()
    # example_lda_document_keywords()
    # example_lda_yearly_topics()
    # example_lda_vs_tfidf()

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("=" * 80 + "\n")
