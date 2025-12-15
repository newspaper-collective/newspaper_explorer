"""
CLI commands for keyword extraction.

Provides commands for extracting keywords using TF-IDF, RAKE, YAKE, and KeyBERT.
For topic modeling, see analyze.topics commands.
"""

import logging
from pathlib import Path
from typing import Optional

import click

from newspaper_explorer.analyze.keywords.keybert import KeyBERTExtractor
from newspaper_explorer.analyze.keywords.rake import RAKEExtractor
from newspaper_explorer.analyze.keywords.tf_idf import TFIDFExtractor
from newspaper_explorer.analyze.keywords.yake import YAKEExtractor
from newspaper_explorer.cli.utils import output
from newspaper_explorer.cli.utils.options import (
    batch_size_option,
    chunk_overlap_option,
    chunk_size_option,
    compile_model_option,
    device_option,
    group_by_option,
    input_file_option,
    limit_option,
    max_length_option,
    min_length_option,
    num_workers_option,
    output_name_option,
    source_option,
    text_column_option,
    top_k_option,
    use_chunking_option,
)


@click.group(name="keywords")
def keywords_group() -> None:
    """Extract keywords using TF-IDF, RAKE, YAKE, or KeyBERT."""
    pass


@keywords_group.command(name="tfidf")
@source_option()
@input_file_option(
    help_text="Custom input parquet file (default: lines.parquet or textblocks.parquet)"
)
@text_column_option()
@click.option(
    "--document-level",
    type=click.Choice(["page", "textblock", "file", "date"], case_sensitive=False),
    default="page",
    help="What constitutes a 'document' (ignored if --group-by is set)",
    show_default=True,
)
@click.option(
    "--group-by",
    type=str,
    help="Custom grouping column(s), comma-separated (e.g., 'year,month'). Overrides --document-level.",
)
@top_k_option(help_text="Number of top keywords to extract per document")
@click.option(
    "--min-df",
    type=int,
    default=2,
    help="Minimum document frequency (ignore terms in fewer documents)",
    show_default=True,
)
@click.option(
    "--max-df",
    type=float,
    default=0.8,
    help="Maximum document frequency as fraction (0-1, ignore very common terms)",
    show_default=True,
)
@click.option(
    "--ngram-range",
    type=str,
    default="1,1",
    help="N-gram range as 'min,max' (e.g., '1,2' for unigrams+bigrams)",
    show_default=True,
)
@click.option(
    "--no-stopwords",
    is_flag=True,
    help="Disable stopword removal",
)
@click.option(
    "--custom-stopwords",
    type=str,
    help="Additional stopwords (comma-separated)",
)
@num_workers_option(
    help_text="Number of CPU workers for parallel extraction (default: auto-detect)"
)
@output_name_option(default="tfidf_keywords")
@limit_option()
def tfidf(
    source: str,
    input_file: str,
    text_column: str,
    document_level: str,
    group_by: str,
    top_k: int,
    min_df: int,
    max_df: float,
    ngram_range: str,
    *,
    no_stopwords: bool,
    custom_stopwords: str,
    num_workers: int,
    output_name: str,
    limit: int,
) -> None:
    """
    Extract keywords using TF-IDF (Term Frequency-Inverse Document Frequency).

    TF-IDF identifies important words by considering both:
    - How often they appear in a document (Term Frequency)
    - How rare they are across all documents (Inverse Document Frequency)

    Words with high TF-IDF scores are frequent in a document but rare overall,
    making them good keywords.

    Preprocessing:
    - TF-IDF does NOT apply preprocessing - users must provide either raw or
      preprocessed data via --input-file and --text-column.
    - For preprocessed text, use: newspaper-explorer data preprocess
      Then specify: --text-column text_processed

    Document Levels:
    - page: One document per newspaper page (RECOMMENDED - best balance)
    - date: One document per publication date (RECOMMENDED - for trends)
    - textblock: One document per text block ([WARNING] Can cause OOM on large datasets)
    - file: One document per XML file (entire issue)

    Examples:

    \b
    Extract keywords from raw text per page:
        newspaper-explorer analyze keywords tfidf --source der_tag

    \b
    Extract keywords from preprocessed text:
        newspaper-explorer analyze keywords tfidf --source der_tag \\
            --input-file data/processed/der_tag/text/preprocessed.parquet \\
            --text-column text_processed

    \b
    Extract keywords per date:
        newspaper-explorer analyze keywords tfidf --source der_tag \\
            --document-level date --top-k 20

    \b
    Custom grouping by year with bigrams:
        newspaper-explorer analyze keywords tfidf --source der_tag \\
            --group-by year --ngram-range 1,2 --top-k 15

    Output:
        Saves to: results/{source}/keywords/{output_name}.parquet
        Columns: grouping columns, keywords (list), scores (list)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output.header("TF-IDF Keyword Extraction")

    output.key_value("Source", source)
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)

    if group_by:
        group_by_list = [g.strip() for g in group_by.split(",")]
        output.key_value("Custom grouping", ", ".join(group_by_list))
    else:
        group_by_list = None
        output.key_value("Document level", document_level.upper())

    output.key_value("Top keywords", top_k)
    output.key_value("Min document frequency", min_df)
    output.key_value("Max document frequency", f"{max_df:.1f}")

    # Parse ngram range
    try:
        ngram_min, ngram_max = map(int, ngram_range.split(","))
        ngram_tuple = (ngram_min, ngram_max)
        if ngram_tuple == (1, 1):
            output.key_value("N-grams", "unigrams only")
        elif ngram_tuple == (1, 2):
            output.key_value("N-grams", "unigrams + bigrams")
        else:
            output.key_value("N-grams", str(ngram_tuple))
    except ValueError:
        output.error(f"Invalid ngram-range format: {ngram_range}")
        output.info("Expected format: 'min,max' (e.g., '1,2')")
        return

    # Parse custom stopwords
    custom_stopwords_list = None
    if custom_stopwords:
        custom_stopwords_list = [w.strip() for w in custom_stopwords.split(",")]
        output.key_value("Custom stopwords", len(custom_stopwords_list))

    use_stopwords = not no_stopwords
    if no_stopwords:
        output.key_value("Stopwords", "disabled")
    else:
        output.key_value("Stopwords", "enabled (German)")

    if limit:
        output.key_value("Limit", f"{limit:,} rows")

    output.info("Extracting keywords...\n")
    try:
        # Initialize extractor
        extractor = TFIDFExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            use_stopwords=use_stopwords,
            custom_stopwords=custom_stopwords_list,
        )

        # Extract keywords
        results_df = extractor.extract_keywords(
            group_by=group_by_list,
            document_level=document_level,
            top_k=top_k,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
            num_workers=num_workers,
            limit=limit,
        )

        # Save results
        output_file = extractor.save_results(results_df, output_name=output_name)

        output.success("Keyword extraction complete!")
        output.key_value("Results saved to", str(output_file))
        output.key_value("Total documents", f"{len(results_df):,}")

        # Show sample
        if len(results_df) > 0:
            output.section("Sample keywords (first 3 documents)")
            sample = results_df.head(3)
            for row in sample.iter_rows(named=True):
                # Show grouping info
                if group_by_list:
                    group_info = ", ".join(
                        [f"{col}={row.get(col, 'N/A')}" for col in group_by_list]
                    )
                    output.info(f"\n{group_info}")
                elif document_level == "page":
                    output.info(
                        f"\nFile: {row.get('filename', 'N/A')}, Page: {row.get('page_number', 'N/A')}"
                    )
                elif document_level == "date":
                    output.info(
                        f"\nDate: {row.get('year', '?')}-{row.get('month', '?')}-{row.get('day', '?')}"
                    )
                else:
                    # Show first grouping column value
                    first_col = next(iter(row.keys()))
                    output.info(f"\n{first_col}: {row.get(first_col, 'N/A')}")

                keywords = row["keywords"][:5]  # Show top 5
                scores = row["scores"][:5]
                for kw, score in zip(keywords, scores):
                    output.info(f"  {kw:20s} {score:.4f}")

    except FileNotFoundError as e:
        output.error(f"Input file not found: {e}")
        output.info("Available options:")
        output.info(f"  1. Run parsing first: newspaper-explorer data parse --source {source}")
        output.info("  2. Specify custom input: --input-file path/to/file.parquet")
        return
    except Exception as e:
        output.error(f"Error during keyword extraction: {e}")
        logging.exception("Full error details:")
        return


@keywords_group.command(name="rake")
@source_option()
@input_file_option(
    help_text="Custom input parquet file (default: textblocks.parquet or lines.parquet)"
)
@text_column_option()
@group_by_option(
    help_text="Column(s) to group by (e.g., 'newspaper_page_id' for pages, 'date' for dates). Default: per page."
)
@top_k_option(help_text="Number of top keyphrases to extract per document")
@min_length_option(default=1)
@max_length_option(default=4, help_text="Maximum number of words in a keyphrase")
@click.option(
    "--use-stopwords/--no-stopwords",
    default=True,
    help="Use German stopwords (recommended)",
)
@batch_size_option(default=1000)
@num_workers_option()
@output_name_option(default="rake_keywords")
def rake(
    source: str,
    input_file: Optional[str],
    text_column: str,
    group_by: Optional[tuple[str, ...]],
    top_k: int,
    min_length: int,
    max_length: int,
    *,
    use_stopwords: bool,
    batch_size: int,
    num_workers: int,
    output_name: str,
) -> None:
    """
    Extract multi-word keyphrases using RAKE (Rapid Automatic Keyword Extraction).

    RAKE uses word co-occurrence and linguistic patterns to extract meaningful
    multi-word phrases from text. It identifies keyphrases by analyzing word
    frequency and co-occurrence within the text.

    Features:
    - Extracts multi-word phrases (not just single words)
    - No training required (rule-based)
    - Fast and language-agnostic
    - Good for extracting domain-specific terminology
    - Parallel processing with batching for large datasets

    Best for: Extracting technical terms, named entities, multi-word concepts

    Examples:

    \b
    Extract keyphrases per page (recommended):
        newspaper-explorer analyze keywords rake --source der_tag

    \b
    Extract longer keyphrases (up to 5 words) with 8 workers:
        newspaper-explorer analyze keywords rake --source der_tag \\
            --max-length 5 --top-k 15 --num-workers 8

    \b
    Extract per date with custom grouping and large batches:
        newspaper-explorer analyze keywords rake --source der_tag \\
            --group-by date --top-k 20 --batch-size 5000

    \b
    Output:
        Saves to: results/{source}/keywords/{output_name}.parquet
        Columns: grouping columns, keywords (list), scores (list)
    """
    output.header("RAKE Keyphrase Extraction")
    output.key_value("Source", source)
    output.key_value("Group by", str(group_by) if group_by else "(page - default)")
    output.key_value("Top keyphrases", top_k)
    output.key_value("Phrase length", f"{min_length}-{max_length} words")
    output.key_value("Use stopwords", use_stopwords)
    output.key_value("Batch size", batch_size)
    if num_workers:
        output.key_value("Workers", num_workers)
    else:
        output.key_value("Workers", "auto-detect (CPU count - 1)")

    try:
        # Initialize extractor
        extractor = RAKEExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            min_phrase_length=min_length,
            max_phrase_length=max_length,
            use_stopwords=use_stopwords,
        )

        # Prepare grouping
        group_by_list = list(group_by) if group_by else None

        # Extract keyphrases with batching and multiprocessing
        output.info("\nExtracting keyphrases...")
        df_keywords = extractor.extract_keyphrases(
            top_k=top_k,
            group_by=group_by_list,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Save results with metadata
        output.info("\nSaving results...")
        output_path = extractor.save_results(df_keywords, output_name=output_name, top_k=top_k)

        output.success("Saved keyphrases")
        output.key_value("Output", str(output_path))
        output.key_value("Total documents", len(df_keywords))
        output.key_value("Columns", ", ".join(df_keywords.columns))

    except FileNotFoundError as e:
        output.error(f"{e}")
        output.info(f"  Run parsing first: newspaper-explorer data parse --source {source}")
        return
    except Exception as e:
        output.error(f"Error during RAKE extraction: {e}")
        logging.exception("Full error details:")
        return


@keywords_group.command(name="yake")
@source_option()
@input_file_option(
    help_text="Custom input parquet file (default: textblocks.parquet or lines.parquet)"
)
@text_column_option()
@group_by_option(
    help_text="Column(s) to group by (e.g., 'newspaper_page_id' for pages, 'date' for dates). Default: per page."
)
@top_k_option(help_text="Number of top keywords to extract per document")
@click.option(
    "--language",
    type=str,
    default="de",
    help="Language code (e.g., 'de' for German, 'en' for English)",
)
@click.option(
    "--max-ngram-size",
    type=int,
    default=3,
    help="Maximum n-gram size (1=words, 2=bigrams, 3=trigrams)",
)
@click.option(
    "--deduplication-threshold",
    type=float,
    default=0.9,
    help="Threshold for removing similar keywords (0-1, higher=less deduplication)",
)
@batch_size_option(default=1000)
@num_workers_option()
@output_name_option(default="yake_keywords")
def yake(
    source: str,
    input_file: Optional[str],
    text_column: str,
    group_by: Optional[tuple[str, ...]],
    top_k: int,
    language: str,
    max_ngram_size: int,
    deduplication_threshold: float,
    batch_size: int,
    num_workers: int,
    output_name: str,
) -> None:
    """
    Extract keywords using YAKE (Yet Another Keyword Extractor).

    YAKE uses statistical text features to extract keywords without requiring
    training data or external corpora. It analyzes word frequency, casing,
    position, and context to identify important keywords.

    Features:
    - Unsupervised (no training needed)
    - Language-independent (specify language for better results)
    - Extracts single words and multi-word expressions
    - Fast extraction

    Best for: General-purpose keyword extraction, balanced results

    Note: YAKE assigns lower scores to more important keywords (inverted scoring)

    Examples:

    \b
    Extract keywords per page (recommended):
        newspaper-explorer analyze keywords yake --source der_tag

    \b
    Extract only single words (unigrams):
        newspaper-explorer analyze keywords yake --source der_tag \\
            --max-ngram-size 1 --top-k 15

    \b
    Extract with stricter deduplication:
        newspaper-explorer analyze keywords yake --source der_tag \\
            --deduplication-threshold 0.7

    \b
    Output:
        Saves to: results/{source}/keywords/{output_name}.parquet
        Columns: grouping columns, keywords (list), scores (list)
    """
    output.header("YAKE Keyword Extraction")
    output.key_value("Source", source)
    output.key_value("Group by", str(group_by) if group_by else "(page - default)")
    output.key_value("Top keywords", top_k)
    output.key_value("Language", language)
    output.key_value("Max n-gram size", max_ngram_size)
    output.key_value("Deduplication threshold", f"{deduplication_threshold:.2f}")
    output.key_value("Batch size", batch_size)
    if num_workers:
        output.key_value("Workers", num_workers)
    else:
        output.key_value("Workers", "auto-detect (CPU count - 1)")

    try:
        # Initialize extractor
        extractor = YAKEExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            language=language,
            max_ngram_size=max_ngram_size,
            deduplication_threshold=deduplication_threshold,
        )

        # Prepare grouping
        group_by_list = list(group_by) if group_by else None

        # Extract keywords
        output.info("\nExtracting keywords...")
        df_keywords = extractor.extract_keywords(
            top_k=top_k,
            group_by=group_by_list,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Save results with metadata
        output.info("\nSaving results...")
        output_path = extractor.save_results(df_keywords, output_name=output_name, top_k=top_k)

        output.success("Saved keywords")
        output.key_value("Output", str(output_path))
        output.key_value("Total documents", len(df_keywords))
        output.key_value("Columns", ", ".join(df_keywords.columns))

    except FileNotFoundError as e:
        output.error(f"{e}")
        output.info(f"  Run parsing first: newspaper-explorer data parse --source {source}")
        return
    except Exception as e:
        output.error(f"Error during YAKE extraction: {e}")
        logging.exception("Full error details:")
        return


@keywords_group.command(name="keybert")
@source_option()
@input_file_option(
    help_text="Custom input parquet file (default: textblocks.parquet or lines.parquet)"
)
@text_column_option()
@group_by_option(
    help_text="Column(s) to group by (e.g., 'newspaper_page_id' for pages, 'date' for dates). Default: per page."
)
@top_k_option(help_text="Number of top keywords to extract per document")
@click.option(
    "--model",
    type=str,
    default="paraphrase-multilingual-MiniLM-L12-v2",
    help="Sentence transformer model (multilingual recommended for German)",
)
@click.option(
    "--min-ngram",
    type=int,
    default=1,
    help="Minimum n-gram size for candidate phrases",
)
@click.option(
    "--max-ngram",
    type=int,
    default=2,
    help="Maximum n-gram size for candidate phrases",
)
@click.option(
    "--diversity",
    type=float,
    default=0.5,
    help="Keyword diversity (0=similar, 1=diverse) using MMR",
)
@click.option(
    "--use-mmr/--no-mmr",
    default=True,
    help="Use Maximal Marginal Relevance for diversity",
)
@output_name_option(default="keybert_keywords")
@batch_size_option(default=32)
@click.option(
    "--max-seq-length",
    type=int,
    default=512,
    help="Maximum sequence length for BERT (truncates longer texts)",
    show_default=True,
)
@device_option()
@use_chunking_option(
    default=True, help_text="Split long texts into chunks to avoid truncation (recommended)"
)
@chunk_size_option(default=400)
@chunk_overlap_option(default=50)
@click.option(
    "--use-multi-gpu",
    type=click.Choice(["auto", "yes", "no"]),
    default="auto",
    help="Use multiple GPUs if available (auto=detect, yes=force, no=disable)",
    show_default=True,
)
@compile_model_option()
@limit_option()
def keybert(
    source: str,
    input_file: Optional[str],
    text_column: str,
    group_by: Optional[tuple[str, ...]],
    top_k: int,
    model: str,
    min_ngram: int,
    max_ngram: int,
    diversity: float,
    *,
    use_mmr: bool,
    output_name: str,
    batch_size: int,
    max_seq_length: int,
    device: Optional[str],
    use_chunking: bool,
    chunk_size: int,
    chunk_overlap: int,
    use_multi_gpu: str,
    compile_model: bool,
    limit: Optional[int],
) -> None:
    """
    Extract keywords using KeyBERT (BERT-based semantic extraction).

    KeyBERT uses BERT embeddings to extract keywords that are semantically
    similar to the document. It computes cosine similarity between document
    embeddings and candidate keyword embeddings.

    Features:
    - Semantic understanding (context-aware)
    - Multilingual support (works well for German)
    - Diversity control via MMR (Maximal Marginal Relevance)
    - Pre-trained models (no training needed)

    Best for: Semantic keyword extraction, capturing meaning and context

    Model options:
    - paraphrase-multilingual-MiniLM-L12-v2 (default, good for German)
    - distiluse-base-multilingual-cased-v2 (larger, more accurate)
    - all-MiniLM-L6-v2 (English only, faster)

    Examples:

    \b
    Extract keywords per page (recommended):
        newspaper-explorer analyze keywords keybert --source der_tag

    \b
    Use larger multilingual model:
        newspaper-explorer analyze keywords keybert --source der_tag \\
            --model distiluse-base-multilingual-cased-v2

    \b
    Extract with high diversity:
        newspaper-explorer analyze keywords keybert --source der_tag \\
            --diversity 0.9 --top-k 15

    \b
    Extract only single words:
        newspaper-explorer analyze keywords keybert --source der_tag \\
            --min-ngram 1 --max-ngram 1

    \b
    Output:
        Saves to: results/{source}/keywords/{output_name}.parquet
        Columns: grouping columns, keywords (list), scores (list)
    """
    output.header("KeyBERT Keyword Extraction")
    output.key_value("Source", source)
    output.key_value("Group by", str(group_by) if group_by else "(page - default)")
    output.key_value("Top keywords", top_k)
    output.key_value("Model", model)
    output.key_value("N-gram range", f"{min_ngram}-{max_ngram}")
    output.key_value("Diversity (MMR)", f"{diversity:.1f}" if use_mmr else "disabled")
    output.key_value("Batch size", batch_size)
    output.key_value("Max sequence length", max_seq_length)
    output.key_value("Chunking", "enabled" if use_chunking else "disabled")
    if use_chunking:
        output.info(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    if compile_model:
        output.info("Model compilation: enabled (torch.compile)")
    output.key_value("Multi-GPU", use_multi_gpu)

    try:
        # Convert multi-GPU string to Optional[bool]
        multi_gpu_setting: Optional[bool]
        if use_multi_gpu == "auto":
            multi_gpu_setting = None
        elif use_multi_gpu == "yes":
            multi_gpu_setting = True
        else:  # "no"
            multi_gpu_setting = False

        # Initialize extractor
        output.info("\nLoading BERT model (this may take a moment)...")
        extractor = KeyBERTExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            model_name=model,
            keyphrase_ngram_range=(min_ngram, max_ngram),
            diversity=diversity,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            device=device,
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_multi_gpu=multi_gpu_setting,
            compile_model=compile_model,
        )

        # Prepare grouping
        group_by_list = list(group_by) if group_by else None

        # Extract keywords
        output.info("Extracting keywords...")
        df_keywords = extractor.extract_keywords(
            top_k=top_k, group_by=group_by_list, use_mmr=use_mmr, limit=limit
        )

        # Save results with metadata
        output_path = extractor.save_results(df_keywords, output_name=output_name, top_k=top_k)

        output.success("Saved keywords")
        output.key_value("Output", str(output_path))
        output.key_value("Total groups", len(df_keywords))
        output.key_value("Columns", ", ".join(df_keywords.columns))

    except FileNotFoundError as e:
        output.error(f"{e}")
        output.info(f"  Run parsing first: newspaper-explorer data parse --source {source}")
        return
    except Exception as e:
        output.error(f"Error during KeyBERT extraction: {e}")
        logging.exception("Full error details:")
        return
