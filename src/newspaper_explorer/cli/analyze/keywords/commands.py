"""
CLI commands for keyword extraction.

Provides commands for extracting keywords using TF-IDF.
"""

import logging
from pathlib import Path

import click

from newspaper_explorer.config.base import get_config


@click.group(name="keywords")
def keywords_group():
    """Extract keywords from newspaper text using TF-IDF."""
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
    "--preprocessing",
    type=str,
    help="Preprocessing steps (comma-separated, e.g., 'normalize,lowercase,remove-punctuation'). Default: basic cleaning.",
)
@click.option(
    "--no-preprocessing",
    is_flag=True,
    help="Skip all preprocessing (use raw text)",
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
    preprocessing: str,
    no_preprocessing: bool,
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

    Document Levels:
    - page: One document per newspaper page (RECOMMENDED - best balance)
    - date: One document per publication date (RECOMMENDED - for trends)
    - textblock: One document per text block (⚠️ Can cause OOM on large datasets)
    - file: One document per XML file (entire issue)

    Examples:

    \b
    Extract keywords per page (recommended):
        newspaper-explorer analyze keywords tfidf --source der_tag

    \b
    Extract keywords per date:
        newspaper-explorer analyze keywords tfidf --source der_tag \\
            --document-level date --top-k 20

    \b
    Custom grouping by year with bigrams:
        newspaper-explorer analyze keywords tfidf --source der_tag \\
            --group-by year --ngram-range 1,2 --top-k 15

    \b
    Use normalized text with advanced preprocessing:
        newspaper-explorer analyze keywords tfidf --source der_tag \\
            --input-file data/processed/der_tag/textblocks_normalized.parquet \\
            --text-column text_normalized --preprocessing normalize,lowercase

    \b
    Skip preprocessing (use raw text):
        newspaper-explorer analyze keywords tfidf --source der_tag \\
            --no-preprocessing

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

    # Parse preprocessing steps
    if no_preprocessing:
        preprocessing_steps = []  # type: ignore
        click.echo("Preprocessing: disabled (raw text)")
    elif preprocessing:
        preprocessing_steps = [s.strip() for s in preprocessing.split(",")]
        click.echo(f"Preprocessing: {', '.join(preprocessing_steps)}")
    else:
        preprocessing_steps = None  # Use defaults
        click.echo("Preprocessing: default (whitespace, lowercase, punctuation)")

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
            preprocessing_steps=preprocessing_steps,
            num_workers=num_workers,
            limit=limit,
        )

        # Save results
        output_file = extractor.save_results(results_df, output_name=output_name)

        click.echo(f"\n{'='*80}")
        click.echo(f"✓ Keyword extraction complete!")
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
