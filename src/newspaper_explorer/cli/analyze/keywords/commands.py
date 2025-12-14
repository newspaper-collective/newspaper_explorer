"""
CLI commands for keyword extraction.

Provides commands for extracting keywords using TF-IDF, RAKE, YAKE, and KeyBERT.
For topic modeling, see analyze.topics commands.
"""

import logging
from pathlib import Path
from typing import Optional

import click

from newspaper_explorer.config.base import get_config


@click.group(name="keywords")
def keywords_group() -> None:
    """Extract keywords using TF-IDF, RAKE, YAKE, or KeyBERT."""
    pass


@keywords_group.command(name="tfidf")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (default: lines.parquet or textblocks.parquet)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
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
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of top keywords to extract per document",
    show_default=True,
)
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
    "--stopwords",
    type=str,
    help="Additional stopwords (comma-separated)",
)
@click.option(
    "--num-workers",
    type=int,
    help="Number of CPU workers for parallel extraction (default: auto-detect)",
)
@click.option(
    "--output-name",
    type=str,
    default="tfidf_keywords",
    help="Base name for output file",
    show_default=True,
)
@click.option("--limit", type=int, help="Limit number of rows to process (for testing)")
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
    no_stopwords: bool,
    stopwords: str,
    num_workers: int,
    output_name: str,
    limit: int,
):
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
    from newspaper_explorer.analyze.keywords.tf_idf import TFIDFExtractor

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    click.echo(f"\n{'='*80}")
    click.echo(f"TF-IDF Keyword Extraction")
    click.echo(f"{'='*80}\n")

    click.echo(f"Source: {source}")
    if input_file:
        click.echo(f"Input file: {input_file}")
    click.echo(f"Text column: {text_column}")

    if group_by:
        group_by_list = [g.strip() for g in group_by.split(",")]
        click.echo(f"Custom grouping: {', '.join(group_by_list)}")
    else:
        group_by_list = None
        click.echo(f"Document level: {document_level.upper()}")

    click.echo(f"Top keywords: {top_k}")
    click.echo(f"Min document frequency: {min_df}")
    click.echo(f"Max document frequency: {max_df}")

    # Parse ngram range
    try:
        ngram_min, ngram_max = map(int, ngram_range.split(","))
        ngram_tuple = (ngram_min, ngram_max)
        if ngram_tuple == (1, 1):
            click.echo("N-grams: unigrams only")
        elif ngram_tuple == (1, 2):
            click.echo("N-grams: unigrams + bigrams")
        else:
            click.echo(f"N-grams: {ngram_tuple}")
    except ValueError:
        click.echo(f"Error: Invalid ngram-range format: {ngram_range}", err=True)
        click.echo("Expected format: 'min,max' (e.g., '1,2')", err=True)
        return

    # Parse custom stopwords
    custom_stopwords = None
    if stopwords:
        custom_stopwords = [w.strip() for w in stopwords.split(",")]
        click.echo(f"Custom stopwords: {len(custom_stopwords)}")

    use_stopwords = not no_stopwords
    if no_stopwords:
        click.echo("Stopwords: disabled")
    else:
        click.echo("Stopwords: enabled (German)")

    if limit:
        click.echo(f"Limit: {limit:,} rows")

    click.echo()

    try:
        # Initialize extractor
        extractor = TFIDFExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            use_stopwords=use_stopwords,
            custom_stopwords=custom_stopwords,
        )

        # Extract keywords
        click.echo("Extracting keywords...\n")
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

        click.echo(f"\n{'='*80}")
        click.echo(f"[OK] Keyword extraction complete!")
        click.echo(f"{'='*80}\n")
        click.echo(f"Results saved to: {output_file}")
        click.echo(f"Total documents: {len(results_df):,}")

        # Show sample
        if len(results_df) > 0:
            click.echo(f"\nSample keywords (first 3 documents):")
            click.echo("-" * 80)
            sample = results_df.head(3)
            for row in sample.iter_rows(named=True):
                # Show grouping info
                if group_by_list:
                    group_info = ", ".join(
                        [f"{col}={row.get(col, 'N/A')}" for col in group_by_list]
                    )
                    click.echo(f"\n{group_info}")
                elif document_level == "page":
                    click.echo(
                        f"\nFile: {row.get('filename', 'N/A')}, Page: {row.get('page_number', 'N/A')}"
                    )
                elif document_level == "date":
                    click.echo(
                        f"\nDate: {row.get('year', '?')}-{row.get('month', '?')}-{row.get('day', '?')}"
                    )
                else:
                    # Show first grouping column value
                    first_col = list(row.keys())[0]
                    click.echo(f"\n{first_col}: {row.get(first_col, 'N/A')}")

                keywords = row["keywords"][:5]  # Show top 5
                scores = row["scores"][:5]
                for kw, score in zip(keywords, scores):
                    click.echo(f"  {kw:20s} {score:.4f}")

        click.echo(f"\n{'='*80}\n")

    except FileNotFoundError as e:
        click.echo(f"\nError: Input file not found: {e}", err=True)
        click.echo("\nAvailable options:", err=True)
        click.echo(
            "  1. Run parsing first: newspaper-explorer data parse --source {source}", err=True
        )
        click.echo("  2. Specify custom input: --input-file path/to/file.parquet", err=True)
        return
    except Exception as e:
        click.echo(f"\nError during keyword extraction: {e}", err=True)
        logging.exception("Full error details:")
        return


@keywords_group.command(name="rake")
@click.option(
    "--source",
    type=str,
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (default: textblocks.parquet or lines.parquet)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
@click.option(
    "--group-by",
    type=str,
    multiple=True,
    default=None,
    help="Column(s) to group by (e.g., 'newspaper_page_id' for pages, 'date' for dates). Can specify multiple times. Default: per page.",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of top keyphrases to extract per document",
)
@click.option(
    "--min-length",
    type=int,
    default=1,
    help="Minimum number of words in a keyphrase",
)
@click.option(
    "--max-length",
    type=int,
    default=4,
    help="Maximum number of words in a keyphrase",
)
@click.option(
    "--use-stopwords/--no-stopwords",
    default=True,
    help="Use German stopwords (recommended)",
)
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    help="Number of documents per batch",
    show_default=True,
)
@click.option(
    "--num-workers",
    type=int,
    help="Number of parallel workers (default: CPU count - 1)",
)
@click.option(
    "--output-name",
    type=str,
    default="rake_keywords",
    help="Output filename (without extension)",
)
def rake(
    source: str,
    input_file: Optional[str],
    text_column: str,
    group_by: Optional[tuple[str, ...]],
    top_k: int,
    min_length: int,
    max_length: int,
    use_stopwords: bool,
    batch_size: int,
    num_workers: int,
    output_name: str,
):
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
    from newspaper_explorer.analyze.keywords.rake import RAKEExtractor

    config = get_config()

    click.echo("\n=== RAKE Keyphrase Extraction ===")
    click.echo(f"Source: {source}")
    click.echo(f"Group by: {group_by or '(page - default)'}")
    click.echo(f"Top keyphrases: {top_k}")
    click.echo(f"Phrase length: {min_length}-{max_length} words")
    click.echo(f"Use stopwords: {use_stopwords}")
    click.echo(f"Batch size: {batch_size}")
    if num_workers:
        click.echo(f"Workers: {num_workers}")
    else:
        click.echo("Workers: auto-detect (CPU count - 1)")

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
        click.echo("\nExtracting keyphrases...")
        df_keywords = extractor.extract_keyphrases(
            top_k=top_k,
            group_by=group_by_list,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Save results with metadata
        click.echo("\nSaving results...")
        output_path = extractor.save_results(df_keywords, output_name=output_name, top_k=top_k)

        click.echo(f"\n[OK] Saved keyphrases to: {output_path}")
        click.echo(f"  Total documents: {len(df_keywords)}")
        click.echo(f"  Columns: {df_keywords.columns}")

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo(
            f"  Run parsing first: newspaper-explorer data parse --source {source}", err=True
        )
        return
    except Exception as e:
        click.echo(f"\nError during RAKE extraction: {e}", err=True)
        logging.exception("Full error details:")
        return


@keywords_group.command(name="yake")
@click.option(
    "--source",
    type=str,
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (default: textblocks.parquet or lines.parquet)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
@click.option(
    "--group-by",
    type=str,
    multiple=True,
    default=None,
    help="Column(s) to group by (e.g., 'newspaper_page_id' for pages, 'date' for dates). Can specify multiple times. Default: per page.",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of top keywords to extract per document",
)
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
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    help="Number of documents per batch for parallel processing",
    show_default=True,
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of worker processes (default: CPU count - 1)",
)
@click.option(
    "--output-name",
    type=str,
    default="yake_keywords",
    help="Output filename (without extension)",
)
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
):
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
    from newspaper_explorer.analyze.keywords.yake import YAKEExtractor

    config = get_config()

    click.echo("\n=== YAKE Keyword Extraction ===")
    click.echo(f"Source: {source}")
    click.echo(f"Group by: {group_by or '(page - default)'}")
    click.echo(f"Top keywords: {top_k}")
    click.echo(f"Language: {language}")
    click.echo(f"Max n-gram size: {max_ngram_size}")
    click.echo(f"Deduplication threshold: {deduplication_threshold}")
    click.echo(f"Batch size: {batch_size}")
    if num_workers:
        click.echo(f"Workers: {num_workers}")
    else:
        click.echo("Workers: auto-detect (CPU count - 1)")

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
        click.echo("\nExtracting keywords...")
        df_keywords = extractor.extract_keywords(
            top_k=top_k,
            group_by=group_by_list,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Save results with metadata
        click.echo("\nSaving results...")
        output_path = extractor.save_results(df_keywords, output_name=output_name, top_k=top_k)

        click.echo(f"\n[OK] Saved keywords to: {output_path}")
        click.echo(f"  Total documents: {len(df_keywords)}")
        click.echo(f"  Columns: {df_keywords.columns}")

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo(
            f"  Run parsing first: newspaper-explorer data parse --source {source}", err=True
        )
        return
    except Exception as e:
        click.echo(f"\nError during YAKE extraction: {e}", err=True)
        logging.exception("Full error details:")
        return


@keywords_group.command(name="keybert")
@click.option(
    "--source",
    type=str,
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (default: textblocks.parquet or lines.parquet)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
@click.option(
    "--group-by",
    type=str,
    multiple=True,
    default=None,
    help="Column(s) to group by (e.g., 'newspaper_page_id' for pages, 'date' for dates). Can specify multiple times. Default: per page.",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of top keywords to extract per document",
)
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
@click.option(
    "--output-name",
    type=str,
    default="keybert_keywords",
    help="Output filename (without extension)",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for GPU processing (increase for better GPU utilization)",
    show_default=True,
)
@click.option(
    "--max-seq-length",
    type=int,
    default=512,
    help="Maximum sequence length for BERT (truncates longer texts)",
    show_default=True,
)
@click.option(
    "--device",
    type=str,
    help="Device to use ('cuda', 'cpu', or None for auto-detect)",
)
@click.option(
    "--use-chunking/--no-chunking",
    default=True,
    help="Split long texts into chunks to avoid truncation (recommended)",
    show_default=True,
)
@click.option(
    "--chunk-size",
    type=int,
    default=400,
    help="Token size for each chunk",
    show_default=True,
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=50,
    help="Overlap between chunks in tokens",
    show_default=True,
)
@click.option(
    "--use-multi-gpu",
    type=click.Choice(["auto", "yes", "no"]),
    default="auto",
    help="Use multiple GPUs if available (auto=detect, yes=force, no=disable)",
    show_default=True,
)
@click.option(
    "--compile-model",
    is_flag=True,
    help="Use torch.compile for model optimization (requires PyTorch 2.0+)",
)
@click.option(
    "--limit",
    type=int,
    help="Limit number of documents to process (for testing)",
)
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
):
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
    from newspaper_explorer.analyze.keywords.keybert import KeyBERTExtractor

    config = get_config()

    click.echo("\n=== KeyBERT Keyword Extraction ===")
    click.echo(f"Source: {source}")
    click.echo(f"Group by: {group_by or '(page - default)'}")
    click.echo(f"Top keywords: {top_k}")
    click.echo(f"Model: {model}")
    click.echo(f"N-gram range: {min_ngram}-{max_ngram}")
    click.echo(f"Diversity (MMR): {diversity if use_mmr else 'disabled'}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Max sequence length: {max_seq_length}")
    click.echo(f"Chunking: {'enabled' if use_chunking else 'disabled'}")
    if use_chunking:
        click.echo(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    if compile_model:
        click.echo(f"Model compilation: enabled (torch.compile)")
    click.echo(f"Multi-GPU: {use_multi_gpu}")

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
        click.echo("\nLoading BERT model (this may take a moment)...")
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
        click.echo("Extracting keywords...")
        df_keywords = extractor.extract_keywords(
            top_k=top_k, group_by=group_by_list, use_mmr=use_mmr, limit=limit
        )

        # Save results with metadata
        output_path = extractor.save_results(df_keywords, output_name=output_name, top_k=top_k)

        click.echo(f"\n[OK] Saved keywords to: {output_path}")
        click.echo(f"  Total groups: {len(df_keywords)}")
        click.echo(f"  Columns: {df_keywords.columns}")

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo(
            f"  Run parsing first: newspaper-explorer data parse --source {source}", err=True
        )
        return
    except Exception as e:
        click.echo(f"\nError during KeyBERT extraction: {e}", err=True)
        logging.exception("Full error details:")
        return
