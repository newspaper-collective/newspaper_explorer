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
    "--filter-pages",
    type=str,
    default=None,
    help="Only process specific page numbers, comma-separated (e.g., '1,2,3' for first 3 pages)",
)
@click.option(
    "--years",
    type=str,
    default=None,
    help="Only process specific years, comma-separated (e.g., '1900,1901,1902')",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Random sample N documents after grouping (for exploration)",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for sampling",
    show_default=True,
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
    filter_pages: str,
    years: str,
    sample_size: int,
    random_seed: int,
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
    
    **Sampling Strategies for Large Corpora (60M+ lines)**:
    
    1. **Page filtering**: --filter-pages 1,2,3 (only first 3 pages per issue)
    2. **Year filtering**: --years 1900,1901 (process specific years)
    3. **Random sampling**: --sample-size 100000 (sample documents)
    4. **UMAP sampling**: --umap-sample-pages 3 (fit on subset, transform all)
    5. **Combined**: Use multiple strategies together
    
    **Recommended workflow for massive datasets**:
    - Process year-by-year (loop with --years)
    - Use --filter-pages 1,2,3 for front pages only
    - Use --umap-sample-pages 3 for memory efficiency
    - Use --sample-size for initial exploration
    
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
    Only process first 3 pages of each issue (huge memory savings):
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --filter-pages 1,2,3

    \b
    Process one year at a time:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --years 1900

    \b
    Combined: first 3 pages + specific years + UMAP sampling:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --filter-pages 1,2,3 --years 1900,1901 \\
            --umap-sample-pages 3

    \b
    Random sample for exploration:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --sample-size 10000

    \b
    Test on subset with larger batches:
        newspaper-explorer analyze topics bertopic \\
            --source der_tag --group-by page_id \\
            --limit 50000 --batch-size 64

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

    # Parse filtering options
    filter_pages_list = None
    if filter_pages:
        filter_pages_list = [int(p.strip()) for p in filter_pages.split(",")]
        click.echo(f"  - Page filter: {filter_pages_list} (only these pages)")

    years_list = None
    if years:
        years_list = [int(y.strip()) for y in years.split(",")]
        click.echo(f"  - Year filter: {years_list} (only these years)")

    if sample_size:
        click.echo(f"  - Random sampling: {sample_size:,} documents (seed={random_seed})")

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
            filter_pages=filter_pages_list,
            years=years_list,
            sample_size=sample_size,
            random_seed=random_seed,
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


@topics_group.command(name="fastopic")
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
    help="Grouping column(s) for aggregation, comma-separated (e.g., 'page_id')",
)
@click.option(
    "--n-topics",
    type=int,
    default=50,
    help="Number of topics to discover",
    show_default=True,
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of topic terms per document",
    show_default=True,
)
@click.option(
    "--doc-embed-model",
    type=str,
    default="paraphrase-multilingual-MiniLM-L12-v2",
    help="SentenceTransformer model for document embeddings",
    show_default=True,
)
@click.option(
    "--vocab-size",
    type=int,
    default=10000,
    help="Maximum vocabulary size",
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
    "--no-stopwords",
    is_flag=True,
    help="Disable German stopword removal",
)
@click.option(
    "--output-name",
    type=str,
    default="fastopic_topics",
    help="Output filename (without extension)",
    show_default=True,
)
@click.option(
    "--limit",
    type=int,
    help="Limit input rows (for testing)",
)
def fastopic(
    source: str,
    input_file: str,
    text_column: str,
    group_by: str,
    n_topics: int,
    top_k: int,
    doc_embed_model: str,
    vocab_size: int,
    min_text_length: int,
    no_stopwords: bool,
    output_name: str,
    limit: int,
):
    """
    Discover topics using FASTopic (Fast, Adaptive, Stable, Transferable).

    FASTopic uses optimal transport between document, topic, and word embeddings
    to efficiently discover topics. It's faster and more stable than BERTopic.

    **Key Features**:
    - Faster training and inference than BERTopic
    - More stable topic assignments
    - Better transferability across domains
    - No clustering required (optimal transport approach)
    
    **Advantages over BERTopic**:
    - 2-3x faster training
    - Deterministic results (no randomness in clustering)
    - Better for smaller datasets (works well with <10k documents)
    - Simpler parameter tuning (just set n_topics)

    **Workflow**:
    1. Load and aggregate documents
    2. Preprocess texts with German tokenization
    3. Train FASTopic model with optimal transport
    4. Extract topic terms and document-topic distributions

    Examples:

    \b
    Discover 50 topics from pages:
        newspaper-explorer analyze topics fastopic \\
            --source der_tag --group-by page_id

    \b
    Discover 30 topics with custom settings:
        newspaper-explorer analyze topics fastopic \\
            --source der_tag --group-by page_id \\
            --n-topics 30 --vocab-size 15000

    \b
    Test on subset:
        newspaper-explorer analyze topics fastopic \\
            --source der_tag --group-by page_id \\
            --limit 5000 --n-topics 20

    Output:
        results/{source}/topics/fastopic_{n_topics}topics_*/
            ├── topics.parquet
            └── topics.json
    """
    from newspaper_explorer.analyze.topics.fastopic import FASTopicExtractor

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    click.echo(f"\n{'='*80}")
    click.echo(f"FASTopic Topic Modeling")
    click.echo(f"{'='*80}\n")
    click.echo(f"Source: {source}")
    click.echo(f"N topics: {n_topics}")
    click.echo(f"Group by: {group_by or 'None (sequential docs)'}")
    click.echo(f"Model: {doc_embed_model}")
    if limit:
        click.echo(f"Limit: {limit:,} rows")
    click.echo(f"\n{'='*80}\n")

    try:
        # Parse group_by
        group_by_list = [col.strip() for col in group_by.split(",")] if group_by else None

        # Initialize extractor
        extractor = FASTopicExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            doc_embed_model=doc_embed_model,
            vocab_size=vocab_size,
            min_text_length=min_text_length,
            use_stopwords=not no_stopwords,
        )

        # Extract topics
        click.echo("Extracting topics...\n")

        results_df = extractor.extract_topics(
            group_by=group_by_list,
            n_topics=n_topics,
            top_k=top_k,
            limit=limit,
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
    except ImportError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo(f"\nInstall FASTopic:", err=True)
        click.echo(f"  pip install fastopic", err=True)
        click.echo(f"  python -m spacy download de_core_news_sm", err=True)
        return
    except Exception as e:
        click.echo(f"\nError during FASTopic processing: {e}", err=True)
        logging.exception("Full error details:")
        return


@topics_group.command(name="bertopic-batch")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
@click.option("--start-year", type=int, required=True, help="First year to process")
@click.option("--end-year", type=int, required=True, help="Last year to process")
@click.option(
    "--filter-pages",
    type=str,
    default="1,2,3",
    help="Page numbers to process, comma-separated",
    show_default=True,
)
@click.option(
    "--umap-sample-pages",
    type=int,
    default=3,
    help="Fit UMAP on first N pages per issue",
    show_default=True,
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=15,
    help="Minimum chunks per topic cluster",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for embedding generation",
    show_default=True,
)
@click.option(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use",
    show_default=True,
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip years that already have output files",
)
@click.option(
    "--stop-on-error",
    is_flag=True,
    help="Stop processing if a year fails (default: continue)",
)
def bertopic_batch(
    source: str,
    start_year: int,
    end_year: int,
    filter_pages: str,
    umap_sample_pages: int,
    min_cluster_size: int,
    batch_size: int,
    num_gpus: int,
    skip_existing: bool,
    stop_on_error: bool,
):
    """
    Process BERTopic year-by-year for large corpora.
    
    This command automates the year-by-year processing workflow for massive
    datasets (60M+ lines). Each year is processed independently with optimal
    memory settings, and results are saved separately.
    
    **Advantages over single run**:
    - 90-95% memory reduction per run
    - Progress is saved (failed years don't affect completed ones)
    - Can be parallelized across machines
    - Results can be merged for temporal analysis
    
    **Recommended for**: Corpora with 10M+ lines
    
    Examples:
    
    \b
    Process all years with default settings:
        newspaper-explorer analyze topics bertopic-batch \\
            --source der_tag --start-year 1900 --end-year 1920
    
    \b
    Custom settings with progress resume:
        newspaper-explorer analyze topics bertopic-batch \\
            --source der_tag --start-year 1900 --end-year 1920 \\
            --filter-pages 1,2,3,4 --min-cluster-size 20 \\
            --skip-existing
    
    \b
    Multi-GPU processing:
        newspaper-explorer analyze topics bertopic-batch \\
            --source der_tag --start-year 1900 --end-year 1920 \\
            --num-gpus 2 --batch-size 128
    
    Output:
        results/{source}/topics/topics_{year}.parquet (one per year)
        results/{source}/topics/topics_{year}.json (metadata)
    
    After completion, use 'merge-yearly' to combine results.
    """
    from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = get_config()
    results_dir = config.results_dir / source / "topics"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Parse filter pages
    filter_pages_list = [int(p.strip()) for p in filter_pages.split(",")]

    total_years = end_year - start_year + 1
    completed = 0
    failed = 0
    skipped = 0

    click.echo(f"\n{'='*80}")
    click.echo(f"BERTopic Year-by-Year Batch Processing")
    click.echo(f"{'='*80}\n")
    click.echo(f"Source: {source}")
    click.echo(f"Years: {start_year} - {end_year} ({total_years} years)")
    click.echo(f"Filter pages: {filter_pages_list}")
    click.echo(f"UMAP sampling: First {umap_sample_pages} pages/issue")
    click.echo(f"Min cluster size: {min_cluster_size}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"GPUs: {num_gpus}")
    click.echo(f"Skip existing: {skip_existing}")
    click.echo(f"Stop on error: {stop_on_error}")
    click.echo(f"{'='*80}\n")

    for year in range(start_year, end_year + 1):
        output_name = f"topics_{year}"
        output_file = results_dir / f"{output_name}.parquet"

        # Check if already exists
        if skip_existing and output_file.exists():
            click.echo(f"[{year}] ⊙ Skipping (already exists): {output_file}")
            skipped += 1
            continue

        click.echo(f"\n[{year}/{end_year}] Processing year {year}...")

        try:
            # Initialize extractor
            extractor = BERTopicExtractor(
                source_name=source,
                num_gpus=num_gpus,
            )

            # Extract topics
            results_df = extractor.extract_topics(
                group_by=["page_id"],
                years=[year],
                filter_pages=filter_pages_list,
                umap_sample_pages=umap_sample_pages,
                min_cluster_size=min_cluster_size,
                batch_size=batch_size,
            )

            if len(results_df) == 0:
                click.echo(f"  ⚠ No results for year {year} - skipping", err=True)
                failed += 1
                if stop_on_error:
                    click.echo("\nStopping due to error (--stop-on-error)", err=True)
                    break
                continue

            # Save results
            extractor.save_results(results_df, output_name=output_name)

            click.echo(f"  ✓ Year {year} completed: {len(results_df):,} documents")
            click.echo(f"  → {output_file}")
            completed += 1

        except Exception as e:
            click.echo(f"  ✗ Year {year} failed: {e}", err=True)
            failed += 1
            if stop_on_error:
                click.echo("\nStopping due to error (--stop-on-error)", err=True)
                logging.exception("Full error details:")
                break
            else:
                click.echo("  Continuing with next year...")

    # Final summary
    click.echo(f"\n{'='*80}")
    click.echo(f"Batch Processing Complete")
    click.echo(f"{'='*80}")
    click.echo(f"Total years: {total_years}")
    click.echo(f"Completed: {completed}")
    click.echo(f"Skipped: {skipped}")
    click.echo(f"Failed: {failed}")
    click.echo(f"{'='*80}\n")

    if completed > 0:
        click.echo("✓ Results saved to:")
        click.echo(f"  {results_dir}/topics_*.parquet")
        click.echo("\nNext step: Merge yearly results")
        click.echo(f"  newspaper-explorer analyze topics merge-yearly \\")
        click.echo(f"    --source {source} --start-year {start_year} --end-year {end_year}")


@topics_group.command(name="merge-yearly")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
@click.option("--start-year", type=int, required=True, help="First year to merge")
@click.option("--end-year", type=int, required=True, help="Last year to merge")
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top topics per year for analysis",
    show_default=True,
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualizations (requires matplotlib)",
)
def merge_yearly(source: str, start_year: int, end_year: int, top_n: int, visualize: bool):
    """
    Merge and analyze yearly BERTopic results.
    
    Combines yearly topic modeling results into a unified dataset and performs
    temporal analysis to track how topics evolve over time.
    
    **Outputs**:
    - Combined dataset with all years
    - Topic evolution analysis (term frequencies by year)
    - Optional visualizations (heatmap, line plots)
    - CSV export for external analysis
    
    Examples:
    
    \b
    Merge all years (after bertopic-batch):
        newspaper-explorer analyze topics merge-yearly \\
            --source der_tag --start-year 1900 --end-year 1920
    
    \b
    With visualizations:
        newspaper-explorer analyze topics merge-yearly \\
            --source der_tag --start-year 1900 --end-year 1920 \\
            --visualize
    
    \b
    Track more topics:
        newspaper-explorer analyze topics merge-yearly \\
            --source der_tag --start-year 1900 --end-year 1920 \\
            --top-n 20
    
    Output:
        results/{source}/topics/analysis/{source}_topics_all_years.parquet
        results/{source}/topics/analysis/{source}_topic_evolution.parquet
        results/{source}/topics/analysis/{source}_topic_evolution.csv
        results/{source}/topics/analysis/{source}_topic_heatmap.png (if --visualize)
        results/{source}/topics/analysis/{source}_topic_trends.png (if --visualize)
    """
    import polars as pl

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = get_config()
    results_dir = config.results_dir / source / "topics"
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*80}")
    click.echo(f"Merging Yearly BERTopic Results")
    click.echo(f"{'='*80}\n")
    click.echo(f"Source: {source}")
    click.echo(f"Years: {start_year} - {end_year}")
    click.echo(f"Results dir: {results_dir}")
    click.echo(f"Output dir: {output_dir}")
    click.echo(f"{'='*80}\n")

    # Load yearly files
    yearly_files = []
    missing_years = []

    for year in range(start_year, end_year + 1):
        file_path = results_dir / f"topics_{year}.parquet"
        if file_path.exists():
            yearly_files.append((year, file_path))
        else:
            missing_years.append(year)

    if not yearly_files:
        click.echo(f"✗ No yearly results found in {results_dir}", err=True)
        click.echo(f"\nRun bertopic-batch first:", err=True)
        click.echo(
            f"  newspaper-explorer analyze topics bertopic-batch "
            f"--source {source} --start-year {start_year} --end-year {end_year}",
            err=True,
        )
        return

    if missing_years:
        click.echo(f"⚠ Missing {len(missing_years)} years: {missing_years}\n")

    click.echo(f"Loading {len(yearly_files)} yearly files...")

    # Load and combine
    dfs = []
    for year, file_path in yearly_files:
        df = pl.read_parquet(file_path)
        df = df.with_columns(pl.lit(year).alias("year"))
        dfs.append(df)
        click.echo(f"  [{year}] {len(df):,} documents")

    combined = pl.concat(dfs)
    click.echo(f"\n✓ Combined: {len(combined):,} documents across {len(yearly_files)} years\n")

    # Analyze topic evolution
    click.echo("Analyzing topic evolution...")
    df_exploded = combined.select(["year", "doc_id", "topic_terms", "scores"]).explode(
        ["topic_terms", "scores"]
    )

    topic_evolution = (
        df_exploded.group_by(["year", "topic_terms"])
        .agg(
            [
                pl.count().alias("frequency"),
                pl.col("scores").mean().alias("avg_score"),
                pl.col("scores").std().alias("std_score"),
            ]
        )
        .sort(["year", "frequency"], descending=[False, True])
    )

    # Get top topics per year
    years = sorted(combined["year"].unique().to_list())

    click.echo(f"\nTop {top_n} topics per year (sample):")
    click.echo("=" * 80)

    for year in years[:5]:  # Show first 5 years
        year_topics = topic_evolution.filter(pl.col("year") == year).head(top_n)
        click.echo(f"\n{year}:")
        click.echo("-" * 40)
        for row in year_topics.iter_rows(named=True):
            term = row["topic_terms"]
            freq = row["frequency"]
            score = row["avg_score"]
            click.echo(f"  {term:20s} | freq: {freq:5d} | score: {score:.3f}")

    if len(years) > 5:
        click.echo(f"\n... and {len(years) - 5} more years (see output files)")

    # Save results
    click.echo(f"\n{'='*80}")
    click.echo("Saving results...")
    click.echo(f"{'='*80}\n")

    combined_file = output_dir / f"{source}_topics_all_years.parquet"
    combined.write_parquet(combined_file)
    click.echo(f"✓ Combined data: {combined_file}")

    evolution_file = output_dir / f"{source}_topic_evolution.parquet"
    topic_evolution.write_parquet(evolution_file)
    click.echo(f"✓ Topic evolution: {evolution_file}")

    evolution_csv = output_dir / f"{source}_topic_evolution.csv"
    topic_evolution.write_csv(evolution_csv)
    click.echo(f"✓ Topic evolution CSV: {evolution_csv}")

    # Generate visualizations
    if visualize:
        click.echo("\nGenerating visualizations...")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Get top terms overall
            top_overall = (
                topic_evolution.group_by("topic_terms")
                .agg(pl.col("frequency").sum().alias("total_frequency"))
                .sort("total_frequency", descending=True)
                .head(20)
            )
            top_terms = top_overall["topic_terms"].to_list()

            df_filtered = topic_evolution.filter(pl.col("topic_terms").is_in(top_terms))
            df_pd = df_filtered.to_pandas()

            # Heatmap
            fig, ax = plt.subplots(figsize=(16, 10))
            pivot = df_pd.pivot(index="topic_terms", columns="year", values="frequency").fillna(0)
            sns.heatmap(
                pivot,
                cmap="YlOrRd",
                cbar_kws={"label": "Document Frequency"},
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title(f"Topic Evolution - {source.upper()}", fontsize=16, pad=20)
            plt.tight_layout()

            heatmap_file = output_dir / f"{source}_topic_heatmap.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
            click.echo(f"✓ Heatmap: {heatmap_file}")
            plt.close()

            # Line plot
            fig, ax = plt.subplots(figsize=(14, 8))
            for term in top_terms[:10]:
                term_data = df_pd[df_pd["topic_terms"] == term]
                ax.plot(term_data["year"], term_data["frequency"], marker="o", label=term)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Document Frequency", fontsize=12)
            ax.set_title(f"Top 10 Topics Over Time - {source.upper()}", fontsize=16, pad=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            trends_file = output_dir / f"{source}_topic_trends.png"
            plt.savefig(trends_file, dpi=300, bbox_inches="tight")
            click.echo(f"✓ Trends plot: {trends_file}")
            plt.close()

        except ImportError:
            click.echo("⚠ matplotlib not installed - skipping visualizations", err=True)
        except Exception as e:
            click.echo(f"⚠ Error generating visualizations: {e}", err=True)

    click.echo(f"\n{'='*80}")
    click.echo("✓ Merge complete!")
    click.echo(f"{'='*80}\n")
    click.echo(f"Results in: {output_dir}")
    click.echo(f"Total documents: {len(combined):,}")
    click.echo(f"Years covered: {len(yearly_files)}")
