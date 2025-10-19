"""
CLI commands for analysis tasks.
Handles entity extraction, topic modeling, and other analysis operations.
"""

import click

# Logging format for CLI - simple messages without timestamps
CLI_LOG_FORMAT = "%(message)s"


@click.group()
def analyze():
    """Run analysis on newspaper data (entities, topics, etc.)."""
    pass


@analyze.command()
@click.option(
    "--source",
    type=str,
    required=True,
    help="Source name (e.g., der_tag)",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to input parquet file (textblocks or sentences)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Column containing text to analyze",
    show_default=True,
)
@click.option(
    "--id-column",
    type=str,
    default="text_block_id",
    help="Column to use as identifier",
    show_default=True,
)
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="Normalize text before extraction",
    show_default=True,
)
@click.option(
    "--model",
    type=str,
    default="urchade/gliner_multi-v2.1",
    help="GLiNER model from Hugging Face Hub",
    show_default=True,
)
@click.option(
    "--labels",
    type=str,
    help="Comma-separated entity labels (default: Person,Organisation,Ereignis,Ort)",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    help="Confidence threshold (0-1)",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for processing",
    show_default=True,
)
@click.option(
    "--min-length",
    type=int,
    default=100,
    help="Minimum text length to process",
    show_default=True,
)
@click.option(
    "--max-length",
    type=int,
    default=500,
    help="Maximum text length (truncate longer)",
    show_default=True,
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["parquet", "json", "both"]),
    default="both",
    help="Output format",
    show_default=True,
)
def extract_entities(
    source,
    input_path,
    text_column,
    id_column,
    normalize,
    model,
    labels,
    threshold,
    batch_size,
    min_length,
    max_length,
    output_format,
):
    """
    Extract named entities from newspaper text using GLiNER.

    This command identifies persons, organizations, events, and locations
    in historical newspaper text.

    \b
    Examples:
      # Extract entities from text blocks
      newspaper-explorer analyze extract-entities \\
          --source der_tag \\
          --input data/processed/der_tag/text/textblocks.parquet

      # Extract from sentences with custom settings
      newspaper-explorer analyze extract-entities \\
          --source der_tag \\
          --input data/processed/der_tag/text/sentences.parquet \\
          --text-column sentence \\
          --batch-size 64 \\
          --threshold 0.6

    \b
    Output:
      Results saved to results/{source}/entities/
      - entities_raw.parquet: All extracted entities with IDs
      - entities_grouped.json: Entities grouped by ID and label
    """
    import logging

    from newspaper_explorer.analysis.entities.extraction import EntityExtractor

    # Configure logging
    logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

    try:
        # Parse labels if provided
        label_list = None
        if labels:
            label_list = [label.strip() for label in labels.split(",")]

        click.echo(f"Extracting entities from: {input_path}")
        click.echo(f"Source: {source}")
        click.echo(f"Model: {model}")
        click.echo("")

        # Initialize extractor
        extractor = EntityExtractor(
            source_name=source,
            model_name=model,
            labels=label_list,
            threshold=threshold,
            batch_size=batch_size,
            min_text_length=min_length,
            max_text_length=max_length,
        )

        # Run extraction
        results = extractor.extract_and_save(
            input_path=input_path,
            text_column=text_column,
            id_column=id_column,
            normalize=normalize,
            output_format=output_format,
        )

        click.echo("\n" + "=" * 60)
        click.echo("✓ Entity extraction complete!")
        click.echo(f"  Total entities: {len(results['entities'])}")
        click.echo(f"  Unique texts: {len(results['serialized'])}")
        click.echo(f"  Output: {extractor.output_dir}")
        click.echo("=" * 60)

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()
