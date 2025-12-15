"""
CLI commands for topic modeling.

Provides commands for topic modeling using LDA, MALLET, BERTopic, FASTopic,
and LLM-based approaches.
"""

import logging
from pathlib import Path

import click

from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor
from newspaper_explorer.analyze.topics.fastopic import FASTopicExtractor
from newspaper_explorer.analyze.topics.lda import LDAExtractor
from newspaper_explorer.analyze.topics.mallet import MALLETExtractor, MALLETNotFoundError
from newspaper_explorer.analyze.topics.utils import (
    generate_topic_visualizations,
    merge_yearly_topics,
    parse_group_by,
    parse_stopwords,
)
from newspaper_explorer.cli.utils import output
from newspaper_explorer.cli.utils.options import (
    batch_size_option,
    input_file_option,
    limit_option,
    min_length_option,
    num_gpus_option,
    num_topics_option,
    source_option,
    text_column_option,
    top_k_option,
)
from newspaper_explorer.config.base import get_config


@click.group(name="topics")
def topics_group() -> None:
    """Discover topics in newspaper text using LDA, BERTopic, or LLMs."""
    pass


@topics_group.command(name="lda")
@source_option()
@input_file_option(
    help_text="Custom input parquet file (default: textblocks.parquet or lines.parquet)"
)
@text_column_option()
@click.option(
    "--mode",
    type=click.Choice(["train", "topics", "documents"], case_sensitive=False),
    default="documents",
    help="Operation: train model, extract topic terms, or assign to documents",
    show_default=True,
)
@num_topics_option(
    default=10,
    help_text="Number of topics to discover/use",
)
@top_k_option(help_text="Number of terms per topic/document")
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
    "--min-doc-count",
    type=int,
    default=5,
    help="Minimum number of documents a term must appear in",
    show_default=True,
)
@click.option(
    "--max-doc-fraction",
    type=float,
    default=0.5,
    help="Maximum fraction of documents a term can appear in (0.0-1.0)",
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
    "--custom-stopwords",
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
@limit_option()
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
    min_doc_count: int,
    max_doc_fraction: float,
    group_by: str,
    *,
    no_stopwords: bool,
    custom_stopwords: str,
    force_retrain: bool,
    output_name: str,
    limit: int,
) -> None:
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

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output.header("LDA Topic Modeling")

    output.key_value("Source", source)
    output.key_value("Mode", mode.upper())
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)
    output.key_value("Number of topics", num_topics)

    # Parse arguments using utility functions
    group_by_list = parse_group_by(group_by)
    if group_by_list:
        output.key_value("Grouping", ", ".join(group_by_list))

    custom_stopwords_list = parse_stopwords(custom_stopwords)
    if custom_stopwords_list:
        output.key_value("Custom stopwords", len(custom_stopwords_list))

    use_stopwords = not no_stopwords
    if no_stopwords:
        output.key_value("Stopwords", "disabled")
    else:
        output.key_value("Stopwords", "enabled (German + English)")

    if limit:
        output.key_value("Limit", f"{limit:,} rows")

    output.info("")

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
            output.info("Training LDA model...")
            output.info("")
            output.section("Parameters")
            output.key_value("Topics", num_topics)
            output.key_value("Passes", passes)
            output.key_value("Iterations", iterations)
            output.key_value("Min doc count", min_doc_count)
            output.key_value("Max doc fraction", f"{max_doc_fraction:.1%}")
            output.info("")

            model_info = extractor.train_model(
                num_topics=num_topics,
                passes=passes,
                iterations=iterations,
                limit=limit,
                group_by=group_by_list,
                no_below=min_doc_count,
                no_above=max_doc_fraction,
                force_retrain=force_retrain,
            )

            output.success("Model training complete!")
            output.key_value("Status", model_info["status"])
            output.key_value("Model saved to", model_info["model_path"])
            if model_info["status"] == "trained":
                output.key_value("Documents", f"{model_info['num_documents']:,}")
                output.key_value("Vocabulary size", f"{model_info['vocabulary_size']:,}")
                output.key_value("Perplexity", f"{model_info['perplexity']:.4f}")
            output.info("")

        elif mode == "topics":
            output.info(f"Extracting representative terms for {num_topics} topics...")
            output.info("")

            results_df = extractor.get_topic_terms(
                top_k=top_k,
                num_topics=num_topics,
            )

            # Save results
            if not output_name:
                output_name = f"lda_topics_{num_topics}"
            output_file = extractor.save_results(results_df, output_name=output_name)

            output.success("Topic term extraction complete!")
            output.key_value("Results saved to", str(output_file))
            output.key_value("Total topics", len(results_df))

            # Show all topics
            output.info("")
            output.section("Topic Terms")
            for row in results_df.iter_rows(named=True):
                topic_id = row["topic_id"]
                topic_terms = row["topic_terms"][:top_k]
                scores = row["scores"][:top_k]

                terms_str = ", ".join(
                    [f"{term}({s:.3f})" for term, s in zip(topic_terms[:10], scores[:10])]
                )
                output.info(f"Topic {topic_id}: {terms_str}")

            output.info("")

        elif mode == "documents":
            output.info("Assigning topics to documents...")
            output.info("")
            output.section("Parameters")
            output.key_value("Topics", num_topics)
            output.key_value("Terms per doc", top_k)
            output.key_value("Min topic prob", f"{min_topic_prob:.1%}")
            output.info("")

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

            output.success("Topic assignment complete!")
            output.key_value("Results saved to", str(output_file))
            output.key_value("Total documents", f"{len(results_df):,}")

            # Show sample
            if len(results_df) > 0:
                output.info("")
                output.section("Sample Document Topics (first 3)")
                sample = results_df.head(3)
                for row in sample.iter_rows(named=True):
                    doc_id = row["doc_id"]
                    topic_terms = row["topic_terms"]
                    scores = row["scores"]
                    topics = row.get("topics", [])
                    topic_probs = row.get("topic_probs", [])

                    output.info(f"Document: {doc_id}")
                    if topics:
                        topic_info = ", ".join(
                            [f"T{t}({p:.2f})" for t, p in zip(topics[:3], topic_probs[:3])]
                        )
                        output.info(f"  Main topics: {topic_info}")
                    output.info(f"  Topic terms: {', '.join(topic_terms[:top_k])}")

            output.info("")

    except FileNotFoundError as e:
        output.error(f"Error: {e}")
        output.info("")
        output.info("Troubleshooting:")
        if "Model not found" in str(e):
            output.info("  Train model first: newspaper-explorer analyze topics lda \\")
            output.info(f"      --source {source} --mode train --num-topics {num_topics}")
        else:
            output.info(f"  Run parsing first: newspaper-explorer data parse --source {source}")
        return
    except Exception as e:
        output.error(f"Error during LDA processing: {e}")
        logging.exception("Full error details:")
        return


@topics_group.command(name="mallet")
@source_option()
@input_file_option(
    help_text="Custom input parquet file (default: textblocks.parquet or lines.parquet)"
)
@text_column_option()
@click.option(
    "--mode",
    type=click.Choice(["train", "topics", "documents"], case_sensitive=False),
    default="documents",
    help="Operation: train model, extract topic terms, or assign to documents",
    show_default=True,
)
@num_topics_option(
    default=20,
    help_text="Number of topics to discover",
)
@top_k_option(help_text="Number of terms per topic/document")
@click.option(
    "--iterations",
    type=int,
    default=1000,
    help="Number of Gibbs sampling iterations",
    show_default=True,
)
@click.option(
    "--optimize-interval",
    type=int,
    default=10,
    help="Iterations between hyperparameter optimization (0 to disable)",
    show_default=True,
)
@click.option(
    "--num-threads",
    type=int,
    default=4,
    help="Number of parallel threads for training",
    show_default=True,
)
@click.option(
    "--alpha",
    type=float,
    default=5.0,
    help="Document-topic prior (alpha)",
    show_default=True,
)
@click.option(
    "--beta",
    type=float,
    default=0.01,
    help="Topic-word prior (beta)",
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
    "--custom-stopwords",
    type=str,
    help="Additional stopwords (comma-separated)",
)
@click.option(
    "--mallet-path",
    type=click.Path(exists=True),
    help="Path to MALLET installation (auto-detected/installed if not specified)",
)
@click.option(
    "--force-retrain",
    is_flag=True,
    help="Force model retraining even if model exists",
)
@click.option(
    "--output-name",
    type=str,
    help="Base name for output file (default: mallet_topics or mallet_documents)",
)
@limit_option()
@click.option(
    "--memory",
    type=int,
    default=8,
    help="Java heap memory in GB for MALLET",
    show_default=True,
)
def mallet(
    source: str,
    input_file: str,
    text_column: str,
    mode: str,
    num_topics: int,
    top_k: int,
    iterations: int,
    optimize_interval: int,
    num_threads: int,
    alpha: float,
    beta: float,
    min_topic_prob: float,
    group_by: str,
    *,
    no_stopwords: bool,
    custom_stopwords: str,
    mallet_path: str,
    force_retrain: bool,
    output_name: str,
    limit: int,
    memory: int,
) -> None:
    """
    Discover topics using MALLET (MAchine Learning for LanguagE Toolkit).

    MALLET provides highly optimized LDA via Gibbs sampling, often producing
    higher quality topics than variational methods (Gensim). It's particularly
    effective for historical texts.

    **Requirements**:
    - Java Runtime Environment (JRE) - MALLET will auto-download if not found

    **Key Advantages over Gensim LDA**:
    - Better topic coherence (Gibbs sampling vs variational Bayes)
    - Automatic hyperparameter optimization
    - Better handling of short documents
    - Generally faster for large corpora

    Modes:
    - train: Train a new MALLET LDA model on the corpus
    - topics: Extract representative terms for each discovered topic
    - documents: Assign topics and their terms to documents

    Workflow:
    1. Train model once: --mode train --num-topics 20
    2. Explore topics: --mode topics --num-topics 20
    3. Assign to documents: --mode documents --num-topics 20

    Examples:

    \b
    Train MALLET model with 20 topics (auto-installs MALLET if needed):
        newspaper-explorer analyze topics mallet --source der_tag \\
            --mode train --num-topics 20 --iterations 1000

    \b
    Extract topic terms:
        newspaper-explorer analyze topics mallet --source der_tag \\
            --mode topics --num-topics 20 --top-k 15

    \b
    Assign topics to documents:
        newspaper-explorer analyze topics mallet --source der_tag \\
            --mode documents --num-topics 20 --top-k 5

    \b
    Train with hyperparameter optimization:
        newspaper-explorer analyze topics mallet --source der_tag \\
            --mode train --num-topics 20 --optimize-interval 20

    \b
    Quick test on sample:
        newspaper-explorer analyze topics mallet --source der_tag \\
            --mode train --num-topics 10 --limit 1000

    Output:
        Models: results/{source}/topics/models/mallet_{num_topics}topics*
        Results: results/{source}/topics/{output_name}.parquet
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output.header("MALLET Topic Modeling")

    output.key_value("Source", source)
    output.key_value("Mode", mode.upper())
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)
    output.key_value("Number of topics", num_topics)

    # Parse arguments using utility functions
    group_by_list = parse_group_by(group_by)
    if group_by_list:
        output.key_value("Grouping", ", ".join(group_by_list))

    custom_stopwords_list = parse_stopwords(custom_stopwords)
    if custom_stopwords_list:
        output.key_value("Custom stopwords", len(custom_stopwords_list))

    use_stopwords = not no_stopwords
    if no_stopwords:
        output.key_value("Stopwords", "disabled")
    else:
        output.key_value("Stopwords", "enabled (German + English)")

    if limit:
        output.key_value("Limit", f"{limit:,} rows")

    output.info("")

    try:
        # Initialize extractor
        extractor = MALLETExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            mallet_path=Path(mallet_path) if mallet_path else None,
            use_stopwords=use_stopwords,
            custom_stopwords=custom_stopwords_list,
        )

        # Execute based on mode
        if mode == "train":
            output.info("Training MALLET model...")
            output.info("")
            output.section("Parameters")
            output.key_value("Topics", num_topics)
            output.key_value("Iterations", iterations)
            output.key_value("Threads", num_threads)
            output.key_value("Alpha", f"{alpha:.2f}")
            output.key_value("Beta", f"{beta:.3f}")
            output.key_value("Optimize interval", optimize_interval)
            output.key_value("Memory", f"{memory}GB")
            output.info("")

            model_info = extractor.train_model(
                num_topics=num_topics,
                iterations=iterations,
                optimize_interval=optimize_interval,
                num_threads=num_threads,
                alpha=alpha,
                beta=beta,
                limit=limit,
                group_by=group_by_list,
                force_retrain=force_retrain,
                memory_gb=memory,
            )

            output.success("Model training complete!")
            output.key_value("Status", model_info["status"])
            if model_info["status"] == "trained":
                output.key_value("Documents", f"{model_info['num_documents']:,}")
                output.key_value("Topic keys", model_info["topic_keys_file"])
                output.key_value("Doc topics", model_info["doc_topics_file"])
                output.key_value("Training time", f"{model_info['training_time_seconds']:.1f}s")
            output.info("")

        elif mode == "topics":
            output.info(f"Extracting representative terms for {num_topics} topics...")
            output.info("")

            results_df = extractor.get_topic_terms(
                top_k=top_k,
                num_topics=num_topics,
            )

            # Save results
            if not output_name:
                output_name = f"mallet_topics_{num_topics}"
            output_file = extractor.save_results(results_df, output_name=output_name)

            output.success("Topic term extraction complete!")
            output.key_value("Results saved to", str(output_file))
            output.key_value("Total topics", len(results_df))

            # Show all topics
            output.info("")
            output.section("Topic Terms")
            for row in results_df.iter_rows(named=True):
                topic_id = row["topic_id"]
                topic_terms = row["topic_terms"][:top_k]
                alpha_val = row.get("alpha", 0)

                output.info(
                    f"Topic {topic_id} (alpha={alpha_val:.4f}): {', '.join(topic_terms[:10])}"
                )

            output.info("")

        elif mode == "documents":
            output.info("Assigning topics to documents...")
            output.info("")
            output.section("Parameters")
            output.key_value("Topics", num_topics)
            output.key_value("Terms per doc", top_k)
            output.key_value("Min topic prob", f"{min_topic_prob:.1%}")
            output.info("")

            results_df = extractor.extract_document_topics(
                top_k=top_k,
                min_topic_prob=min_topic_prob,
                num_topics=num_topics,
            )

            # Save results
            if not output_name:
                output_name = f"mallet_documents_{num_topics}topics"
            output_file = extractor.save_results(results_df, output_name=output_name)

            output.success("Topic assignment complete!")
            output.key_value("Results saved to", str(output_file))
            output.key_value("Total documents", f"{len(results_df):,}")

            # Show sample
            if len(results_df) > 0:
                output.info("")
                output.section("Sample Document Topics (first 3)")
                sample = results_df.head(3)
                for row in sample.iter_rows(named=True):
                    doc_id = row["doc_id"]
                    topic_terms = row["topic_terms"]
                    topics = row.get("topics", [])
                    topic_probs = row.get("topic_probs", [])

                    output.info(f"Document: {doc_id}")
                    if topics:
                        topic_info = ", ".join(
                            [f"T{t}({p:.2f})" for t, p in zip(topics[:3], topic_probs[:3])]
                        )
                        output.info(f"  Main topics: {topic_info}")
                    output.info(f"  Topic terms: {', '.join(topic_terms[:top_k])}")

            output.info("")

    except MALLETNotFoundError as e:
        output.error(f"MALLET Installation Error: {e}")
        output.info("")
        output.info("To install Java:")
        output.info("  Ubuntu/Debian: sudo apt install default-jre")
        output.info("  macOS: brew install openjdk")
        output.info("  Windows: Download from https://adoptium.net/")
        return
    except FileNotFoundError as e:
        output.error(f"Error: {e}")
        output.info("")
        output.info("Troubleshooting:")
        if "Model not found" in str(e):
            output.info("  Train model first: newspaper-explorer analyze topics mallet \\")
            output.info(f"      --source {source} --mode train --num-topics {num_topics}")
        else:
            output.info(f"  Run parsing first: newspaper-explorer data parse --source {source}")
        return
    except Exception as e:
        output.error(f"Error during MALLET processing: {e}")
        logging.exception("Full error details:")
        return


@topics_group.command(name="bertopic")
@source_option()
@input_file_option(help_text="Custom input parquet file (default: textblocks.parquet or lines.parquet)")
@text_column_option()
@click.option(
    "--group-by",
    type=str,
    help="Grouping column(s), comma-separated (e.g., 'page_id', 'date', 'year,month')",
)
@num_topics_option(
    default=None,
    help_text="Target number of topics (None = automatic discovery)",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=10,
    help="Minimum chunks per topic cluster",
    show_default=True,
)
@top_k_option(
    default=5,
    help_text="Number of topic terms per document",
)
@batch_size_option(
    default=32,
)
@click.option(
    "--embedding-model",
    type=str,
    default="paraphrase-multilingual-MiniLM-L12-v2",
    help="SentenceTransformer model for embeddings",
    show_default=True,
)
@min_length_option(
    default=100,
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
@num_gpus_option(
    default=1,
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
@limit_option()
def bertopic(
    source: str,
    input_file: str,
    text_column: str,
    group_by: str,
    num_topics: int,
    min_cluster_size: int,
    top_k: int,
    batch_size: int,
    embedding_model: str,
    min_length: int,
    chunk_sentences: int,
    *,
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
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    output.header("BERTopic Global Topic Modeling")

    output.section("Parameters")
    output.key_value("Source", source)
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)
    output.key_value("Embedding model", embedding_model)
    output.key_value("Target topics", num_topics or "automatic discovery")
    output.key_value("Min cluster size", min_cluster_size)
    output.key_value("Terms per document", top_k)
    output.key_value("Batch size", batch_size)
    output.key_value("Chunk sentences", chunk_sentences)
    output.key_value("Min text length", f"{min_length} chars")
    output.key_value("Stopwords", "disabled" if no_stopwords else "enabled (German)")
    gpu_mode = "multi-GPU" if num_gpus > 1 else "single GPU"
    output.key_value("GPUs", f"{num_gpus} ({gpu_mode})")

    # Parse filtering options
    filter_pages_list = None
    if filter_pages:
        filter_pages_list = [int(p.strip()) for p in filter_pages.split(",")]
        output.key_value("Page filter", f"{filter_pages_list} (only these pages)")

    years_list = None
    if years:
        years_list = [int(y.strip()) for y in years.split(",")]
        output.key_value("Year filter", f"{years_list} (only these years)")

    if sample_size:
        output.key_value("Random sampling", f"{sample_size:,} documents (seed={random_seed})")

    if umap_sample_pages:
        output.key_value(
            "UMAP sampling", f"first {umap_sample_pages} pages/issue (memory optimization)"
        )

    # Parse group_by
    group_by_list = parse_group_by(group_by)
    if group_by_list:
        output.key_value("Group by", ", ".join(group_by_list))

    if limit:
        output.key_value("Limit", f"{limit:,} rows")

    try:
        # Initialize extractor
        extractor = BERTopicExtractor(
            source_name=source,
            input_file=Path(input_file) if input_file else None,
            text_column=text_column,
            embedding_model=embedding_model,
            min_text_length=min_length,
            chunk_sentences=chunk_sentences,
            use_stopwords=not no_stopwords,
            num_gpus=num_gpus,
        )

        # Extract topics
        output.info("Extracting topics...")

        results_df = extractor.extract_topics(
            group_by=group_by_list,
            n_topics=num_topics,
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
            output.error("No results generated - check logs for details")
            return

        # Save results
        output_file = extractor.save_results(results_df, output_name=output_name, top_k=top_k)

        output.success("Topic extraction complete!")
        output.key_value("Results saved to", str(output_file))

        # Get metadata file path
        metadata_file = str(output_file).replace(".parquet", ".json")
        output.key_value("Metadata saved to", metadata_file)
        output.key_value("Total documents", f"{len(results_df):,}")

        # Show sample
        if len(results_df) > 0:
            output.section("Sample document topics (first 3)")
            sample = results_df.head(3)
            for row in sample.iter_rows(named=True):
                doc_id = row["doc_id"]
                topics = row["topics"]
                topic_probs = row["topic_probs"]
                topic_terms = row["topic_terms"]

                topic_info = ", ".join(
                    [f"T{t}({p:.2f})" for t, p in zip(topics[:3], topic_probs[:3])]
                )
                output.info(f"\nDocument: {doc_id}")
                output.info(f"  Main topics: {topic_info}")
                output.info(f"  Topic terms: {', '.join(topic_terms[:top_k])}")
                if len(row["scores"]) > 0:
                    output.info(
                        f"  Scores: {', '.join([f'{s:.3f}' for s in row['scores'][:top_k]])}"
                    )

    except FileNotFoundError as e:
        output.error(str(e))
        output.info("Troubleshooting:")
        output.info(f"  Run parsing first: newspaper-explorer data parse --source {source}")
        return
    except Exception as e:
        output.error(f"Error during BERTopic processing: {e}")
        logging.exception("Full error details:")
        return


@topics_group.command(name="fastopic")
@source_option()
@input_file_option(
    help_text="Custom input parquet file (default: textblocks.parquet or lines.parquet)"
)
@text_column_option()
@click.option(
    "--group-by",
    type=str,
    help="Grouping column(s) for aggregation, comma-separated (e.g., 'page_id')",
)
@num_topics_option(
    default=50,
    help_text="Number of topics to discover",
)
@top_k_option(
    default=10,
    help_text="Number of topic terms per document",
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
@min_length_option(
    default=100,
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
@limit_option()
def fastopic(
    source: str,
    input_file: str,
    text_column: str,
    group_by: str,
    num_topics: int,
    top_k: int,
    doc_embed_model: str,
    vocab_size: int,
    min_length: int,
    *,
    no_stopwords: bool,
    output_name: str,
    limit: int,
) -> None:
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
    - Simpler parameter tuning (just set num_topics)

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
        results/{source}/topics/fastopic_{num_topics}topics_*/
            ├── topics.parquet
            └── topics.json
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    output.header("FASTopic Topic Modeling")
    output.key_value("Source", source)
    output.key_value("N topics", str(num_topics))
    output.key_value("Group by", group_by or "None (sequential docs)")
    output.key_value("Model", doc_embed_model)
    if limit:
        output.key_value("Limit", f"{limit:,} rows")

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
            min_text_length=min_length,
            use_stopwords=not no_stopwords,
        )

        # Extract topics
        output.section("Extracting topics")

        results_df = extractor.extract_topics(
            group_by=group_by_list,
            n_topics=num_topics,
            top_k=top_k,
            limit=limit,
        )

        if len(results_df) == 0:
            output.error("No results generated - check logs for details")
            return

        # Save results
        output_file = extractor.save_results(results_df, output_name=output_name, top_k=top_k)

        output.section("Results")
        output.success("Topic extraction complete")
        output.key_value("Saved to", str(output_file))

        # Show sample
        if len(results_df) > 0:
            output.section("Sample document topics (first 3)")
            sample = results_df.head(3)
            for row in sample.iter_rows(named=True):
                doc_id = row["doc_id"]
                topics = row["topics"]
                topic_probs = row["topic_probs"]
                topic_terms = row["topic_terms"]

                topic_info = ", ".join(
                    [f"T{t}({p:.2f})" for t, p in zip(topics[:3], topic_probs[:3])]
                )
                output.info(f"\nDocument: {doc_id}")
                output.info(f"  Main topics: {topic_info}")
                output.info(f"  Topic terms: {', '.join(topic_terms[:top_k])}")
                if len(row["scores"]) > 0:
                    output.info(
                        f"  Scores: {', '.join([f'{s:.3f}' for s in row['scores'][:top_k]])}"
                    )

    except FileNotFoundError as e:
        output.error(str(e))
        output.info("\nTroubleshooting:")
        output.info(f"  Run parsing first: newspaper-explorer data parse --source {source}")
        return
    except ImportError as e:
        output.error(str(e))
        output.info("\nInstall FASTopic:")
        output.info("  pip install fastopic")
        output.info("  python -m spacy download de_core_news_sm")
        return
    except (OSError, RuntimeError) as e:
        output.error(f"Error during FASTopic processing: {e}")
        logging.exception("Full error details:")
        return


@topics_group.command(name="bertopic-batch")
@source_option()
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
@batch_size_option(
    default=64,
)
@num_gpus_option(
    default=1,
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
    *,
    skip_existing: bool,
    stop_on_error: bool,
) -> None:
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

    output.header("BERTopic Year-by-Year Batch Processing")
    output.key_value("Source", source)
    output.key_value("Years", f"{start_year} - {end_year} ({total_years} years)")
    output.key_value("Filter pages", str(filter_pages_list))
    output.key_value("UMAP sampling", f"First {umap_sample_pages} pages/issue")
    output.key_value("Min cluster size", str(min_cluster_size))
    output.key_value("Batch size", str(batch_size))
    output.key_value("GPUs", str(num_gpus))
    output.key_value("Skip existing", str(skip_existing))
    output.key_value("Stop on error", str(stop_on_error))

    for year in range(start_year, end_year + 1):
        output_name = f"topics_{year}"
        output_file = results_dir / f"{output_name}.parquet"

        # Check if already exists
        if skip_existing and output_file.exists():
            output.info(f"[{year}] ⊙ Skipping (already exists): {output_file}")
            skipped += 1
            continue

        output.section(f"[{year}/{end_year}] Processing year {year}")

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
                output.info(f"  [WARNING] No results for year {year} - skipping")
                failed += 1
                if stop_on_error:
                    output.error("Stopping due to error (--stop-on-error)")
                    break
                continue

            # Save results
            extractor.save_results(results_df, output_name=output_name)

            output.success(f"Year {year} completed: {len(results_df):,} documents")
            output.info(f"  → {output_file}")
            completed += 1

        except (OSError, RuntimeError) as e:
            output.error(f"Year {year} failed: {e}")
            failed += 1
            if stop_on_error:
                output.error("Stopping due to error (--stop-on-error)")
                logging.exception("Full error details:")
                break
            output.info("  Continuing with next year...")

    # Final summary
    output.header("Batch Processing Complete")
    output.key_value("Total years", str(total_years))
    output.key_value("Completed", str(completed))
    output.key_value("Skipped", str(skipped))
    output.key_value("Failed", str(failed))

    if completed > 0:
        output.success("Results saved to:")
        output.info(f"  {results_dir}/topics_*.parquet")
        output.info("\nNext step: Merge yearly results")
        output.info("  newspaper-explorer analyze topics merge-yearly \\")
        output.info(f"    --source {source} --start-year {start_year} --end-year {end_year}")


@topics_group.command(name="merge-yearly")
@source_option()
@click.option("--start-year", type=int, required=True, help="First year to merge")
@click.option("--end-year", type=int, required=True, help="Last year to merge")
@top_k_option(
    default=10,
    help_text="Number of top topics per year for analysis",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualizations (requires matplotlib)",
)
def merge_yearly(
    source: str,
    start_year: int,
    end_year: int,
    top_k: int,
    *,
    visualize: bool,
) -> None:
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
            --top-k 20

    Output:
        results/{source}/topics/analysis/{source}_topics_all_years.parquet
        results/{source}/topics/analysis/{source}_topic_evolution.parquet
        results/{source}/topics/analysis/{source}_topic_evolution.csv
        results/{source}/topics/analysis/{source}_topic_heatmap.png (if --visualize)
        results/{source}/topics/analysis/{source}_topic_trends.png (if --visualize)
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = get_config()
    results_dir = config.results_dir / source / "topics"
    output_dir = results_dir / "analysis"

    output.header("Merging Yearly BERTopic Results")
    output.key_value("Source", source)
    output.key_value("Years", f"{start_year} - {end_year}")
    output.key_value("Results dir", str(results_dir))
    output.key_value("Output dir", str(output_dir))

    # Merge yearly topics using utility function
    try:
        result = merge_yearly_topics(
            source=source,
            start_year=start_year,
            end_year=end_year,
            results_dir=results_dir,
            output_dir=output_dir,
            top_n=top_k,
        )
    except FileNotFoundError as e:
        output.error(str(e))
        output.info("\nRun bertopic-batch first:")
        output.info(
            f"  newspaper-explorer analyze topics bertopic-batch "
            f"--source {source} --start-year {start_year} --end-year {end_year}"
        )
        return

    # Display warnings for missing years
    if result["missing_years"]:
        output.info(f"Missing {len(result['missing_years'])} years: {result['missing_years']}")

    # Show loading status
    max_years_to_display = 5
    output.section(f"Loading {len(result['yearly_files'])} yearly files")
    for year, _file_path in result["yearly_files"]:
        count = len(result["combined"].filter(result["combined"]["year"] == year))
        output.info(f"  [{year}] {count:,} documents")

    combined_count = len(result["combined"])
    output.success(
        f"Combined: {combined_count:,} documents across {len(result['yearly_files'])} years"
    )

    # Show sample topic evolution
    topic_evolution = result["topic_evolution"]
    years = sorted(result["combined"]["year"].unique().to_list())

    output.section(f"Top {top_k} topics per year (sample)")
    for year in years[:max_years_to_display]:
        year_topics = topic_evolution.filter(topic_evolution["year"] == year).head(top_k)
        output.info(f"\n{year}:")
        output.info("-" * 40)
        for row in year_topics.iter_rows(named=True):
            term = row["topic_terms"]
            freq = row["frequency"]
            score = row["avg_score"]
            output.info(f"  {term:20s} | freq: {freq:5d} | score: {score:.3f}")

    if len(years) > max_years_to_display:
        output.info(f"\n... and {len(years) - max_years_to_display} more years (see output files)")

    # Show saved files
    output.section("Saved Results")
    output.key_value("Combined data", str(result["output_files"]["combined"]))
    output.key_value("Topic evolution", str(result["output_files"]["evolution"]))
    output.key_value("Evolution CSV", str(result["output_files"]["evolution_csv"]))

    # Generate visualizations if requested
    if visualize:
        output.section("Generating visualizations")
        try:
            viz_result = generate_topic_visualizations(
                topic_evolution=topic_evolution,
                source=source,
                output_dir=output_dir,
                top_n_terms=20,
            )
            output.key_value("Heatmap", str(viz_result["heatmap"]))
            output.key_value("Trends plot", str(viz_result["trends"]))
        except ImportError:
            output.info("matplotlib not installed - skipping visualizations")
        except (OSError, RuntimeError) as e:
            output.info(f"Error generating visualizations: {e}")

    # Final summary
    output.header("Merge Complete")
    output.key_value("Results in", str(output_dir))
    output.key_value("Total documents", f"{combined_count:,}")
    output.key_value("Years covered", str(len(result["yearly_files"])))
