"""
CLI commands for entity extraction.

Provides commands for extracting named entities using GLiNER or LLM methods.
"""

import logging
import click

from newspaper_explorer.config.base import get_config


@click.group(name="entities")
def entities_group():
    """Extract named entities from newspaper text."""
    pass


@entities_group.command(name="list-models")
def list_models():
    """List available GLiNER models."""
    from newspaper_explorer.analyze.entities.gliner import GLINER_MODELS

    click.echo("\nAvailable GLiNER Models:")
    click.echo("=" * 80)

    for short_name, info in GLINER_MODELS.items():
        default_marker = " (default)" if info.get("default") else ""
        click.echo(f"\n{short_name}{default_marker}")
        click.echo(f"  Model ID: {info['model_id']}")
        click.echo(f"  Size: {info['size']}")
        click.echo(f"  Max tokens: {info['max_tokens']}")
        click.echo(f"  Description: {info['description']}")

    click.echo("\n" + "=" * 80)
    click.echo("\nUsage:")
    click.echo("  newspaper-explorer analyze entities gliner --source der_tag \\")
    click.echo("    --labels 'Person,Organisation,Ort' --model xxl-v2.5")
    click.echo("\nOr use full Hugging Face model ID:")
    click.echo("  --model 'urchade/gliner_multi-v2.1'")
    click.echo("")


@entities_group.command(name="gliner")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
@click.option("--labels", type=str, required=True, help="Comma-separated entity labels to extract")
@click.option(
    "--input-type",
    type=click.Choice(["textblocks", "lines"], case_sensitive=False),
    default="textblocks",
    help="Input data type (textblocks recommended for better context)",
    show_default=True,
)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (overrides --input-type, e.g., textblocks_normalized.parquet)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
@click.option(
    "--id-column",
    type=str,
    help="Name of ID column (auto-detected: text_block_id or line_id)",
)
@click.option(
    "--model",
    type=str,
    default="multi-v2.1",
    help="GLiNER model (multi-v2.1, xxl-v2.5, small-v2.1, large-v2.1, or full HF ID)",
    show_default=True,
)
@click.option(
    "--threshold", type=float, default=0.2, help="Confidence threshold (0-1)", show_default=True
)
@click.option(
    "--batch-size", type=int, default=32, help="Batch size for processing", show_default=True
)
@click.option("--min-length", type=int, default=10, help="Minimum text length", show_default=True)
@click.option(
    "--max-length",
    type=int,
    default=1000,
    help="Maximum text length (~1000 chars = ~200 tokens)",
    show_default=True,
)
@click.option("--limit", type=int, help="Limit number of rows to process")
@click.option(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for parallel processing (auto-detects available GPUs)",
    show_default=True,
)
def gliner(
    source,
    labels,
    input_type,
    input_file,
    text_column,
    id_column,
    model,
    threshold,
    batch_size,
    min_length,
    max_length,
    limit,
    num_gpus,
):
    """Extract named entities using GLiNER model (fast, local, offline)."""
    from newspaper_explorer.analyze.entities.gliner import GLiNEREntityExtractor
    from pathlib import Path

    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    try:
        label_list = [label.strip() for label in labels.split(",")]

        # Determine input file path
        if input_file:
            # Custom file provided
            input_path = Path(input_file)
            # Auto-detect ID column if not provided
            if not id_column:
                import polars as pl

                df_sample = pl.read_parquet(input_path, n_rows=1)
                if "text_block_id" in df_sample.columns:
                    id_col = "text_block_id"
                elif "line_id" in df_sample.columns:
                    id_col = "line_id"
                else:
                    click.echo("Error: Cannot auto-detect ID column. Use --id-column", err=True)
                    raise click.Abort()
            else:
                id_col = id_column
        else:
            # Use input-type to determine default file
            if input_type == "textblocks":
                input_path = Path("data") / "processed" / source / "text" / "textblocks.parquet"
                id_col = id_column or "text_block_id"
            else:  # lines
                input_path = Path("data") / "raw" / source / "text" / f"{source}_lines.parquet"
                id_col = id_column or "line_id"

            if not input_path.exists():
                click.echo(f"Error: Input file not found: {input_path}", err=True)
                if input_type == "textblocks":
                    click.echo(
                        "Run 'newspaper-explorer data aggregate --source {source}' first", err=True
                    )
                raise click.Abort()

        click.echo(f"Extracting entities from source: {source}")
        click.echo(f"Input: {input_path}")
        click.echo(f"Text column: {text_column}, ID column: {id_col}")
        click.echo(f"Model: {model}")
        click.echo(f"Labels: {', '.join(label_list)}")
        click.echo("")

        extractor = GLiNEREntityExtractor(
            source_name=source,
            model_name=model,
            labels=label_list,
            threshold=threshold,
            batch_size=batch_size,
            min_text_length=min_length,
            max_text_length=max_length,
        )

        results = extractor.extract_and_save(
            source_parquet=input_path,
            limit=limit,
            text_column=text_column,
            id_column=id_col,
            num_gpus=num_gpus,
        )

        click.echo(f"\n✓ Complete! Method ID: {results['metadata']['analysis_id']}")
        click.echo(f"  Entities: {len(results['results_df'])}")
        click.echo(f"  Output: {results['output_dir']}")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@entities_group.command(name="gliner2")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
@click.option("--labels", type=str, required=True, help="Comma-separated entity labels to extract")
@click.option(
    "--label-descriptions",
    type=str,
    help='JSON dict with label descriptions for better accuracy (e.g., \'{"Person":"Names of people"}\')',
)
@click.option(
    "--input-type",
    type=click.Choice(["textblocks", "lines"], case_sensitive=False),
    default="textblocks",
    help="Input data type (textblocks recommended for better context)",
    show_default=True,
)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (overrides --input-type)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
@click.option(
    "--id-column",
    type=str,
    help="Name of ID column (auto-detected: text_block_id or line_id)",
)
@click.option(
    "--model",
    type=str,
    default="fastino/gliner2-base-0207",
    help="GLiNER2 model (base=205M, large=340M)",
    show_default=True,
)
@click.option(
    "--threshold", type=float, default=0.5, help="Confidence threshold (0-1)", show_default=True
)
@click.option(
    "--batch-size", type=int, default=32, help="Batch size for processing", show_default=True
)
@click.option("--min-length", type=int, default=100, help="Minimum text length", show_default=True)
@click.option(
    "--max-length",
    type=int,
    default=2500,
    help="Maximum text length before chunking",
    show_default=True,
)
@click.option("--limit", type=int, help="Limit number of rows (for testing)")
def gliner2(
    source,
    labels,
    label_descriptions,
    input_type,
    input_file,
    text_column,
    id_column,
    model,
    threshold,
    batch_size,
    min_length,
    max_length,
    limit,
):
    """Extract named entities using GLiNER2 (optimized CPU inference, 205M params)."""
    from newspaper_explorer.analyze.entities.gliner2 import GLiNER2EntityExtractor
    from pathlib import Path
    import json as json_module

    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    try:
        # Parse labels
        label_list = [label.strip() for label in labels.split(",")]

        # Parse label descriptions if provided
        labels_dict: dict[str, str | dict] = {}
        if label_descriptions:
            try:
                parsed_dict = json_module.loads(label_descriptions)
                if not isinstance(parsed_dict, dict):
                    click.echo("Error: --label-descriptions must be a JSON dict", err=True)
                    raise click.Abort()
                labels_dict = parsed_dict
            except json_module.JSONDecodeError as e:
                click.echo(f"Error parsing --label-descriptions JSON: {e}", err=True)
                raise click.Abort()
        else:
            # Use German descriptions by default for better accuracy
            labels_dict = {
                label: f"{label} in German historical newspaper text" for label in label_list
            }

        # Determine input file path
        if input_file:
            # Custom file provided
            input_path = Path(input_file)
            # Auto-detect ID column if not provided
            if not id_column:
                import polars as pl

                df_sample = pl.read_parquet(input_path, n_rows=1)
                if "text_block_id" in df_sample.columns:
                    id_col = "text_block_id"
                elif "line_id" in df_sample.columns:
                    id_col = "line_id"
                else:
                    click.echo("Error: Cannot auto-detect ID column. Use --id-column", err=True)
                    raise click.Abort()
            else:
                id_col = id_column
        else:
            # Use input-type to determine default file
            if input_type == "textblocks":
                input_path = Path("data") / "processed" / source / "text" / "textblocks.parquet"
                id_col = id_column or "text_block_id"
            else:  # lines
                input_path = Path("data") / "raw" / source / "text" / f"{source}_lines.parquet"
                id_col = id_column or "line_id"

            if not input_path.exists():
                click.echo(f"Error: Input file not found: {input_path}", err=True)
                if input_type == "textblocks":
                    click.echo(
                        f"Run 'newspaper-explorer data aggregate --source {source}' first", err=True
                    )
                raise click.Abort()

        click.echo(f"Extracting entities from source: {source}")
        click.echo(f"Input: {input_path}")
        click.echo(f"Text column: {text_column}, ID column: {id_col}")
        click.echo(f"Model: {model} (GLiNER2, CPU-optimized)")
        click.echo(f"Labels: {', '.join(label_list)}")
        click.echo(f"Using descriptions: Yes")
        click.echo("")

        extractor = GLiNER2EntityExtractor(
            source_name=source,
            model_name=model,
            labels=labels_dict,
            threshold=threshold,
            batch_size=batch_size,
            min_text_length=min_length,
            max_text_length=max_length,
        )

        results = extractor.extract_and_save(
            source_parquet=input_path,
            limit=limit,
            text_column=text_column,
            id_column=id_col,
        )

        click.echo(f"\n✓ Complete! Method ID: {results['metadata']['analysis_id']}")
        click.echo(f"  Entities: {len(results['results_df'])}")
        click.echo(f"  Output: {results['output_dir']}")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@entities_group.command(name="llm")
@click.option("--source", type=str, required=True, help="Source name")
@click.option(
    "--input-type",
    type=click.Choice(["textblocks", "lines"], case_sensitive=False),
    default="textblocks",
    help="Input data type (textblocks recommended for better context)",
    show_default=True,
)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (overrides --input-type)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
@click.option(
    "--id-column",
    type=str,
    help="Name of ID column (auto-detected)",
)
@click.option("--model", type=str, default="gpt-4o-mini", help="LLM model", show_default=True)
@click.option(
    "--temperature", type=float, default=0.3, help="Sampling temperature", show_default=True
)
@click.option("--max-tokens", type=int, default=2000, help="Maximum tokens", show_default=True)
@click.option(
    "--min-length", type=int, default=100, help="Minimum text length in chars", show_default=True
)
@click.option(
    "--max-length",
    type=int,
    default=8000,
    help="Maximum text length before chunking (chars)",
    show_default=True,
)
@click.option("--batch-size", type=int, default=10, help="Batch size", show_default=True)
@click.option("--limit", type=int, help="Limit number of rows")
def llm(
    source,
    input_type,
    input_file,
    text_column,
    id_column,
    model,
    temperature,
    max_tokens,
    min_length,
    max_length,
    batch_size,
    limit,
):
    """Extract named entities using LLM (high quality, requires API)."""
    from newspaper_explorer.analyze.entities.llm import LLMEntityExtractor
    from pathlib import Path

    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    try:
        # Determine input file path
        if input_file:
            # Custom file provided
            input_path = Path(input_file)
            # Auto-detect ID column if not provided
            if not id_column:
                import polars as pl

                df_sample = pl.read_parquet(input_path, n_rows=1)
                if "text_block_id" in df_sample.columns:
                    id_col = "text_block_id"
                elif "line_id" in df_sample.columns:
                    id_col = "line_id"
                else:
                    click.echo("Error: Cannot auto-detect ID column. Use --id-column", err=True)
                    raise click.Abort()
            else:
                id_col = id_column
        else:
            # Use input-type to determine default file
            if input_type == "textblocks":
                input_path = Path("data") / "processed" / source / "text" / "textblocks.parquet"
                id_col = id_column or "text_block_id"
            else:  # lines
                input_path = Path("data") / "raw" / source / "text" / f"{source}_lines.parquet"
                id_col = id_column or "line_id"

            if not input_path.exists():
                click.echo(f"Error: Input file not found: {input_path}", err=True)
                if input_type == "textblocks":
                    click.echo(
                        "Run 'newspaper-explorer data aggregate --source {source}' first", err=True
                    )
                raise click.Abort()

        click.echo(f"Extracting entities from source: {source}")
        click.echo(f"Input: {input_path}")
        click.echo(f"Text column: {text_column}, ID column: {id_col}")
        click.echo(f"Model: {model}")
        click.echo(f"Text length: min={min_length}, max={max_length} chars")
        click.echo("")

        extractor = LLMEntityExtractor(
            source_name=source,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=batch_size,
            min_text_length=min_length,
            max_text_length=max_length,
        )

        results = extractor.extract_and_save(
            source_parquet=input_path,
            limit=limit,
            text_column=text_column,
            id_column=id_col,
        )

        click.echo(f"\n✓ Complete! Method ID: {results['metadata']['analysis_id']}")
        click.echo(f"  Entities: {len(results['results_df'])}")
        click.echo(f"  Output: {results['output_dir']}")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@entities_group.command(name="list-results")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
def list_results(source: str):
    """
    List all available entity extraction results for a source.

    Shows method IDs, models used, entity counts, and timestamps.

    Example:
        newspaper-explorer analyze entities list-results --source der_tag
    """
    from newspaper_explorer.analyze.entities.utils import list_entity_results

    try:
        results = list_entity_results(source)

        if not results:
            click.echo(f"\nNo entity extraction results found for source '{source}'")
            return

        click.echo(f"\nEntity Extraction Results for '{source}':")
        click.echo("=" * 100)
        click.echo(f"{'Method ID':<50} {'Type':<10} {'Model':<20} {'Entities':<10} {'Lines':<8}")
        click.echo("=" * 100)

        for result in results:
            method_id = result["method_id"]
            method_type = result["method_type"]
            model = result["model"][:18] + ".." if len(result["model"]) > 20 else result["model"]
            entity_count = result["entity_count"]
            line_count = result["line_count"]

            click.echo(
                f"{method_id:<50} {method_type:<10} {model:<20} {entity_count:<10} {line_count:<8}"
            )

        click.echo("=" * 100)
        click.echo(f"\nTotal: {len(results)} results")
        click.echo("\nUse 'compare' command to compare two results:")
        click.echo(f"  newspaper-explorer analyze entities compare --source {source} \\")
        click.echo("    --method-1 <method_id_1> --method-2 <method_id_2>")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@entities_group.command(name="compare")
@click.option("--source", type=str, required=True, help="Source name (e.g., der_tag)")
@click.option("--method-1", type=str, required=True, help="First method ID to compare")
@click.option("--method-2", type=str, required=True, help="Second method ID to compare")
@click.option(
    "--detailed/--summary",
    default=False,
    help="Show detailed comparison (default: summary only)",
)
def compare_results(source: str, method_1: str, method_2: str, detailed: bool):
    """
    Compare two existing entity extraction results.

    This command loads and compares two previously saved entity extraction
    results without re-running the extraction. Use 'list-results' to see
    available method IDs.

    Example:
        # List available results
        newspaper-explorer analyze entities list-results --source der_tag

        # Compare two results
        newspaper-explorer analyze entities compare --source der_tag \\
            --method-1 llm_gpt-4o-mini_20241103_143022 \\
            --method-2 gliner_multi-v2-1_20241103_143145
    """
    from newspaper_explorer.analyze.entities.utils import (
        compare_existing_results,
        print_comparison_report,
    )

    try:
        click.echo(f"\nComparing entity extraction results for '{source}':")
        click.echo(f"  Method 1: {method_1}")
        click.echo(f"  Method 2: {method_2}")
        click.echo("")

        # Load and compare
        comparison_data = compare_existing_results(source, method_1, method_2)

        # Print report
        if detailed:
            print_comparison_report(comparison_data)
        else:
            # Summary only
            comparisons = comparison_data["comparisons"]
            click.echo("\n### Statistics ###")
            click.echo(comparisons["statistics"])

            click.echo("\n### Entity Type Distribution ###")
            click.echo(comparisons["type_distribution"].head(20))

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\nUse 'list-results' to see available method IDs:")
        click.echo(f"  newspaper-explorer analyze entities list-results --source {source}")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()
