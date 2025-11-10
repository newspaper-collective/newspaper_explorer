"""
CLI commands for topic modeling.

Provides commands for topic modeling using LDA, BERTopic, and LLM-based approaches.
"""

import logging
from pathlib import Path

import click

from newspaper_explorer.config.base import get_config


@click.group(name="topics")
def topics_group():
    """Discover topics in newspaper text using LDA, BERTopic, or LLMs."""
    pass


@topics_group.command(name="lda")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
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
    "--mode",
    type=click.Choice(["train", "topics", "documents"], case_sensitive=False),
    default="documents",
    help="Operation: train model, extract topic terms, or assign to documents",
    show_default=True,
)
@click.option(
    "--num-topics",
    type=int,
    default=10,
    help="Number of topics to discover/use",
    show_default=True,
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of terms per topic/document",
    show_default=True,
)
@click.option(
    "--passes",
    type=int,
    default=10,
    help="Number of training passes through corpus",
    show_default=True,
)
@click.option(
    "--iterations",
    type=int,
    default=400,
    help="Maximum iterations per pass",
    show_default=True,
)
@click.option(
    "--min-topic-prob",
    type=float,
    default=0.1,
    help="Minimum topic probability for document assignment",
    show_default=True,
)
@click.option(
    "--no-below",
    type=int,
    default=5,
    help="Filter tokens appearing in < N documents",
    show_default=True,
)
@click.option(
    "--no-above",
    type=float,
    default=0.5,
    help="Filter tokens appearing in > fraction of documents",
    show_default=True,
)
@click.option(
    "--group-by",
    type=str,
    help="Grouping column(s) for aggregation, comma-separated (e.g., 'year,month')",
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
    "--force-retrain",
    is_flag=True,
    help="Force model retraining even if model exists",
)
@click.option(
    "--output-name",
    type=str,
    help="Base name for output file (default: lda_topics or lda_documents)",
)
@click.option("--limit", type=int, help="Limit number of rows to process (for testing)")
def lda(
    source: str,
    input_file: str,
    text_column: str,
    mode: str,
    num_topics: int,
    top_k: int,
    passes: int,
    iterations: int,
    min_topic_prob: float,
    no_below: int,
    no_above: float,
    group_by: str,
    no_stopwords: bool,
    stopwords: str,
    force_retrain: bool,
    output_name: str,
    limit: int,
):
    """
    Discover topics using LDA (Latent Dirichlet Allocation).

    LDA is a probabilistic topic modeling technique that discovers latent themes
    in your corpus. Each topic is represented by a distribution of terms, and each
    document is a mixture of topics.

    This is TOPIC MODELING, not keyword extraction. Use `analyze keywords` commands
    for extracting salient keywords from documents.

    Modes:
    - train: Train a new LDA model on the corpus
    - topics: Extract representative terms for each discovered topic
    - documents: Assign topics and their terms to documents

    Workflow:
    1. Train model once: --mode train --num-topics 20
    2. Explore topics: --mode topics --num-topics 20
    3. Assign to documents: --mode documents --num-topics 20

    Examples:

    \b
    Train LDA model with 20 topics:
        newspaper-explorer analyze topics lda --source der_tag \\
            --mode train --num-topics 20 --passes 15

    \b
    Extract topic terms:
        newspaper-explorer analyze topics lda --source der_tag \\
            --mode topics --num-topics 20 --top-k 15

    \b
    Assign topics to documents:
        newspaper-explorer analyze topics lda --source der_tag \\
            --mode documents --num-topics 20 --top-k 5

    \b
    Train on aggregated yearly data:
        newspaper-explorer analyze topics lda --source der_tag \\
            --mode train --num-topics 15 --group-by year

    \b
    Quick test on sample:
        newspaper-explorer analyze topics lda --source der_tag \\
            --mode train --num-topics 10 --limit 1000

    Output:
        Models: results/{source}/topics/models/lda_{num_topics}topics.model
        Results: results/{source}/topics/{output_name}.parquet
    """
    from newspaper_explorer.analyze.topics.lda import LDAExtractor

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    click.echo(f"\n{'='*80}")
    click.echo(f"LDA Topic Modeling")
    click.echo(f"{'='*80}\n")

    click.echo(f"Source: {source}")
    click.echo(f"Mode: {mode.upper()}")
    if input_file:
        click.echo(f"Input file: {input_file}")
    click.echo(f"Text column: {text_column}")
    click.echo(f"Number of topics: {num_topics}")

    # Parse group_by
    group_by_list = None
    if group_by:
        group_by_list = [g.strip() for g in group_by.split(",")]
        click.echo(f"Grouping: {', '.join(group_by_list)}")

    # Parse custom stopwords
    custom_stopwords = None
    if stopwords:
        custom_stopwords = [w.strip() for w in stopwords.split(",")]
        click.echo(f"Custom stopwords: {len(custom_stopwords)}")

    use_stopwords = not no_stopwords
    if no_stopwords:
        click.echo("Stopwords: disabled")
    else:
        click.echo("Stopwords: enabled (German + English)")

    if limit:
        click.echo(f"Limit: {limit:,} rows")

    click.echo()

    try:
        # Initialize extractor
        extractor = LDAExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            use_stopwords=use_stopwords,
            custom_stopwords=custom_stopwords,
        )

        # Execute based on mode
        if mode == "train":
            click.echo("Training LDA model...\n")
            click.echo(f"Parameters:")
            click.echo(f"  - Topics: {num_topics}")
            click.echo(f"  - Passes: {passes}")
            click.echo(f"  - Iterations: {iterations}")
            click.echo(f"  - Min document freq: {no_below}")
            click.echo(f"  - Max document freq: {no_above}")
            click.echo()

            model_info = extractor.train_model(
                num_topics=num_topics,
                passes=passes,
                iterations=iterations,
                limit=limit,
                group_by=group_by_list,
                no_below=no_below,
                no_above=no_above,
                force_retrain=force_retrain,
            )

            click.echo(f"\n{'='*80}")
            click.echo(f"✓ Model training complete!")
            click.echo(f"{'='*80}\n")
            click.echo(f"Status: {model_info['status']}")
            click.echo(f"Model saved to: {model_info['model_path']}")
            if model_info["status"] == "trained":
                click.echo(f"Documents: {model_info['num_documents']:,}")
                click.echo(f"Vocabulary size: {model_info['vocabulary_size']:,}")
                click.echo(f"Perplexity: {model_info['perplexity']:.4f}")
            click.echo(f"\n{'='*80}\n")

        elif mode == "topics":
            click.echo(f"Extracting representative terms for {num_topics} topics...\n")

            results_df = extractor.get_topic_terms(
                top_k=top_k,
                num_topics=num_topics,
            )

            # Save results
            if not output_name:
                output_name = f"lda_topics_{num_topics}"
            output_file = extractor.save_results(results_df, output_name=output_name)

            click.echo(f"\n{'='*80}")
            click.echo(f"✓ Topic term extraction complete!")
            click.echo(f"{'='*80}\n")
            click.echo(f"Results saved to: {output_file}")
            click.echo(f"Total topics: {len(results_df)}")

            # Show all topics
            click.echo(f"\nTopic terms:")
            click.echo("-" * 80)
            for row in results_df.iter_rows(named=True):
                topic_id = row["topic_id"]
                topic_terms = row["topic_terms"][:top_k]
                scores = row["scores"][:top_k]

                click.echo(f"\nTopic {topic_id}:")
                terms_str = ", ".join(
                    [f"{term}({s:.3f})" for term, s in zip(topic_terms[:10], scores[:10])]
                )
                click.echo(f"  {terms_str}")

            click.echo(f"\n{'='*80}\n")

        elif mode == "documents":
            click.echo(f"Assigning topics to documents...\n")
            click.echo(f"Parameters:")
            click.echo(f"  - Topics: {num_topics}")
            click.echo(f"  - Terms per doc: {top_k}")
            click.echo(f"  - Min topic prob: {min_topic_prob}")
            click.echo()

            results_df = extractor.extract_topic_based_document_terms(
                top_k=top_k,
                min_topic_prob=min_topic_prob,
                num_topics=num_topics,
                limit=limit,
                group_by=group_by_list,
            )

            # Save results
            if not output_name:
                output_name = f"lda_documents_{num_topics}topics"
            output_file = extractor.save_results(results_df, output_name=output_name)

            click.echo(f"\n{'='*80}")
            click.echo(f"✓ Topic assignment complete!")
            click.echo(f"{'='*80}\n")
            click.echo(f"Results saved to: {output_file}")
            click.echo(f"Total documents: {len(results_df):,}")

            # Show sample
            if len(results_df) > 0:
                click.echo(f"\nSample document topics (first 3):")
                click.echo("-" * 80)
                sample = results_df.head(3)
                for row in sample.iter_rows(named=True):
                    doc_id = row["doc_id"]
                    topic_terms = row["topic_terms"]
                    scores = row["scores"]
                    topics = row.get("topics", [])
                    topic_probs = row.get("topic_probs", [])

                    click.echo(f"\nDocument: {doc_id}")
                    if topics:
                        topic_info = ", ".join(
                            [f"T{t}({p:.2f})" for t, p in zip(topics[:3], topic_probs[:3])]
                        )
                        click.echo(f"  Main topics: {topic_info}")
                    click.echo(f"  Topic terms: {', '.join(topic_terms[:top_k])}")

            click.echo(f"\n{'='*80}\n")

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\nTroubleshooting:", err=True)
        if "Model not found" in str(e):
            click.echo(f"  Train model first: newspaper-explorer analyze topics lda \\", err=True)
            click.echo(f"      --source {source} --mode train --num-topics {num_topics}", err=True)
        else:
            click.echo(
                f"  Run parsing first: newspaper-explorer data parse --source {source}", err=True
            )
        return
    except Exception as e:
        click.echo(f"\nError during LDA processing: {e}", err=True)
        logging.exception("Full error details:")
        return


@topics_group.command(name="bertopic")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
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
    help="Grouping column(s), comma-separated (e.g., 'page_id', 'date', 'year,month')",
)
@click.option(
    "--n-topics",
    type=int,
    default=None,
    help="Target number of topics (None = automatic discovery)",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=10,
    help="Minimum chunks per topic cluster",
    show_default=True,
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of topic terms per document",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for embedding generation",
    show_default=True,
)
@click.option(
    "--embedding-model",
    type=str,
    default="paraphrase-multilingual-MiniLM-L12-v2",
    help="SentenceTransformer model for embeddings",
    show_default=True,
)
@click.option(
    "--min-text-length",
    type=int,
    default=100,
    help="Minimum text length in characters",
    show_default=True,
)
@click.option(
    "--chunk-sentences",
    type=int,
    default=5,
    help="Sentences per chunk",
    show_default=True,
)
@click.option(
    "--no-stopwords",
    is_flag=True,
    help="Disable German stopword removal",
)
@click.option(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for embedding (1=single, >1=multi-GPU parallel)",
    show_default=True,
)
@click.option(
    "--umap-sample-pages",
    type=int,
    default=None,
    help="Fit UMAP on first N pages of each issue (e.g., 3) to reduce memory usage by 10-100x",
)
@click.option(
    "--output-name",
    type=str,
    default="bertopic_topics",
    help="Output filename (without extension)",
    show_default=True,
)
@click.option(
    "--limit",
    type=int,
    help="Limit input rows (for testing)",
)
def bertopic(
    source: str,
    input_file: str,
    text_column: str,
    group_by: str,
    n_topics: int,
    min_cluster_size: int,
    top_k: int,
    batch_size: int,
    embedding_model: str,
    min_text_length: int,
    chunk_sentences: int,
    no_stopwords: bool,
    num_gpus: int,
    umap_sample_pages: int,
    output_name: str,
    limit: int,
):
    """
    Discover topics using BERTopic (global semantic topic modeling).

    **Global Topics Approach**:
    Fits a single BERTopic model on the entire corpus to discover global topics,
    then assigns documents to these topics. This approach:
    
    - Enables temporal analysis (same topic space across all documents)
    - Is 10-100x faster than per-document modeling
    - Produces better topic quality (learns from entire corpus)
    - Supports multi-GPU for embedding generation
    
    Workflow:
    1. Chunk all documents into sentences
    2. Embed all chunks in batches (multi-GPU supported)
    3. Fit BERTopic ONCE to discover global topics
    4. Assign each document to discovered topics

    Examples:

    \b
    Discover topics from all pages:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id

    \b
    Discover 20 topics with custom settings:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --n-topics 20 --min-cluster-size 15

    \b
    Test on subset with larger batches:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --limit 50000 --batch-size 64

    \b
    Temporal analysis by date:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by date

    \b
    Large corpus with memory optimization (fit UMAP on first 3 pages):
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --umap-sample-pages 3

    Output:
        results/{source}/topics/{output_name}.parquet
        results/{source}/topics/{output_name}.json
    """
    from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    click.echo(f"\n{'='*80}")
    click.echo(f"BERTopic Global Topic Modeling")
    click.echo(f"{'='*80}\n")

    click.echo(f"Parameters:")
    click.echo(f"  - Source: {source}")
    if input_file:
        click.echo(f"  - Input file: {input_file}")
    click.echo(f"  - Text column: {text_column}")
    click.echo(f"  - Embedding model: {embedding_model}")
    click.echo(f"  - Target topics: {n_topics or 'automatic discovery'}")
    click.echo(f"  - Min cluster size: {min_cluster_size}")
    click.echo(f"  - Terms per document: {top_k}")
    click.echo(f"  - Batch size: {batch_size}")
    click.echo(f"  - Chunk sentences: {chunk_sentences}")
    click.echo(f"  - Min text length: {min_text_length} chars")
    click.echo(f"  - Stopwords: {'disabled' if no_stopwords else 'enabled (German)'}")
    click.echo(f"  - GPUs: {num_gpus} ({'multi-GPU' if num_gpus > 1 else 'single GPU'})")
    if umap_sample_pages:
        click.echo(
            f"  - UMAP sampling: first {umap_sample_pages} pages/issue (memory optimization)"
        )

    # Parse group_by
    group_by_list = None
    if group_by:
        group_by_list = [g.strip() for g in group_by.split(",")]
        click.echo(f"  - Group by: {', '.join(group_by_list)}")

    if limit:
        click.echo(f"  - Limit: {limit:,} rows")

    click.echo()

    try:
        # Initialize extractor
        extractor = BERTopicExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            embedding_model=embedding_model,
            min_text_length=min_text_length,
            chunk_sentences=chunk_sentences,
            use_stopwords=not no_stopwords,
            num_gpus=num_gpus,
        )

        # Extract topics
        click.echo("Extracting topics...\n")

        results_df = extractor.extract_topics(
            group_by=group_by_list,
            n_topics=n_topics,
            min_cluster_size=min_cluster_size,
            top_k=top_k,
            limit=limit,
            batch_size=batch_size,
            umap_sample_pages=umap_sample_pages,
        )

        if len(results_df) == 0:
            click.echo("\nNo results generated - check logs for details", err=True)
            return

        # Save results
        output_file = extractor.save_results(results_df, output_name=output_name, top_k=top_k)

        click.echo(f"\n{'='*80}")
        click.echo(f"✓ Topic extraction complete!")
        click.echo(f"{'='*80}\n")
        click.echo(f"Results saved to: {output_file}")

        # Get metadata file path
        metadata_file = str(output_file).replace(".parquet", ".json")
        click.echo(f"Metadata saved to: {metadata_file}")
        click.echo(f"Total documents: {len(results_df):,}")

        # Show sample
        if len(results_df) > 0:
            click.echo(f"\nSample document topics (first 3):")
            click.echo("-" * 80)
            sample = results_df.head(3)
            for row in sample.iter_rows(named=True):
                doc_id = row["doc_id"]
                topics = row["topics"]
                topic_probs = row["topic_probs"]
                topic_terms = row["topic_terms"]

                topic_info = ", ".join(
                    [f"T{t}({p:.2f})" for t, p in zip(topics[:3], topic_probs[:3])]
                )
                click.echo(f"\nDocument: {doc_id}")
                click.echo(f"  Main topics: {topic_info}")
                click.echo(f"  Topic terms: {', '.join(topic_terms[:top_k])}")
                if len(row["scores"]) > 0:
                    click.echo(
                        f"  Scores: {', '.join([f'{s:.3f}' for s in row['scores'][:top_k]])}"
                    )

        click.echo(f"\n{'='*80}\n")

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo(f"\nTroubleshooting:", err=True)
        click.echo(
            f"  Run parsing first: newspaper-explorer data parse --source {source}", err=True
        )
        return
    except Exception as e:
        click.echo(f"\nError during BERTopic processing: {e}", err=True)
        logging.exception("Full error details:")
        return
