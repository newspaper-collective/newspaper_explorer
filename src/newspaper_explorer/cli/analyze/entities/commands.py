"""
CLI commands for entity extraction.

Provides commands for extracting named entities using GLiNER or LLM methods.
"""

import logging
from typing import Optional
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

        click.echo(f"\n[OK] Complete! Method ID: {results['metadata'].analysis_id}")
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

        click.echo(f"\n[OK] Complete! Method ID: {results['metadata'].analysis_id}")
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

        click.echo(f"\n[OK] Complete! Method ID: {results['metadata'].analysis_id}")
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


@entities_group.command(name="network")
@click.option("--source", type=str, default="der_tag", help="Source name (e.g., der_tag)")
@click.option(
    "--method-id",
    type=str,
    default=None,
    help="Entity extraction method ID (uses most recent if not specified)",
)
@click.option(
    "--connection-type",
    type=click.Choice(["page", "issue", "date"], case_sensitive=False),
    default="page",
    help="How to define connections (page=same page, issue=same issue, date=same date)",
)
@click.option(
    "--min-cooccur",
    type=int,
    default=2,
    help="Minimum co-occurrences for edge",
)
@click.option(
    "--entity-types",
    type=str,
    default=None,
    help="Comma-separated entity types to include (e.g., 'person,organization')",
)
@click.option(
    "--top-n",
    type=int,
    default=20,
    help="Show top N most connected entities",
)
def network_stats(
    source: str,
    method_id: Optional[str],
    connection_type: str,
    min_cooccur: int,
    entity_types: Optional[str],
    top_n: int,
):
    """
    Build and analyze entity co-occurrence network.

    Creates a network where entities are connected if they appear together
    on the same page/issue/date, then shows network statistics and most
    connected entities.

    Example:
        # Build network from page co-occurrences
        newspaper-explorer analyze entities network --source der_tag \\
            --connection-type page --min-cooccur 2

        # Filter to only persons and organizations
        newspaper-explorer analyze entities network --source der_tag \\
            --entity-types "person,organization" --top-n 30
    """
    from newspaper_explorer.analyze.entities.network import EntityNetworkAnalyzer

    try:
        click.echo(f"\nBuilding entity network for '{source}'")
        if method_id:
            click.echo(f"Method ID: {method_id}")
        else:
            click.echo("Using most recent entity extraction")
        click.echo("")

        # Parse entity types
        entity_type_list = None
        if entity_types:
            entity_type_list = [t.strip() for t in entity_types.split(",")]
            click.echo(f"Filtering to entity types: {entity_type_list}")

        # Build network
        analyzer = EntityNetworkAnalyzer(source_name=source, method_id=method_id)
        stats = analyzer.build_network(
            connection_type=connection_type,
            min_co_occurrences=min_cooccur,
            entity_types=entity_type_list,
        )

        # Print network stats
        click.echo("\n" + "=" * 80)
        click.echo("NETWORK STATISTICS")
        click.echo("=" * 80)
        click.echo(f"Nodes (entities): {stats['nodes']}")
        click.echo(f"Edges (connections): {stats['edges']}")
        click.echo(f"Connection type: {stats['connection_type']}")
        click.echo(f"Min co-occurrences: {stats['min_co_occurrences']}")
        click.echo(f"Avg connections per entity: {stats['avg_connections_per_entity']:.2f}")

        # Get comprehensive stats
        full_stats = analyzer.get_network_stats()
        click.echo(f"\nMax degree: {full_stats['max_degree']}")
        click.echo(f"Min degree: {full_stats['min_degree']}")
        click.echo(f"Total co-occurrences: {full_stats['total_co_occurrences']}")

        # Show top entities
        click.echo("\n" + "=" * 80)
        click.echo(f"TOP {top_n} MOST CONNECTED ENTITIES (by degree)")
        click.echo("=" * 80)
        click.echo(f"{'Entity':<40} {'Connections':<15} {'Total Co-occur':<15}")
        click.echo("-" * 80)

        top_entities = analyzer.get_top_entities(n=top_n, metric="degree")
        for entity, degree in top_entities:
            total_cooccur = analyzer.entity_metadata[entity]["total_co_occurrences"]
            click.echo(f"{entity:<40} {int(degree):<15} {total_cooccur:<15}")

        click.echo("\n" + "=" * 80)
        click.echo("\nUse 'find-path' command to find connections between entities.")
        click.echo("")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@entities_group.command(name="find-path")
@click.option("--source", type=str, default="der_tag", help="Source name")
@click.option("--from", "source_entity", type=str, required=True, help="Source entity name")
@click.option("--to", "target_entity", type=str, required=True, help="Target entity name")
@click.option(
    "--method-id",
    type=str,
    default=None,
    help="Entity extraction method ID (uses most recent if not specified)",
)
@click.option(
    "--connection-type",
    type=click.Choice(["page", "issue", "date"], case_sensitive=False),
    default="page",
    help="How to define connections",
)
@click.option(
    "--min-cooccur",
    type=int,
    default=1,
    help="Minimum co-occurrences for edge",
)
@click.option(
    "--max-degrees",
    type=int,
    default=6,
    help="Maximum degrees of separation to search",
)
@click.option(
    "--show-details",
    is_flag=True,
    help="Show detailed connection information",
)
def find_path(
    source: str,
    source_entity: str,
    target_entity: str,
    method_id: Optional[str],
    connection_type: str,
    min_cooccur: int,
    max_degrees: int,
    show_details: bool,
):
    """
    Find shortest path between two entities (Six Degrees of Separation).

    Searches for connection paths between entities through their co-appearances
    in the newspaper, showing how entities are related.

    Example:
        # Find path between two entities
        newspaper-explorer analyze entities find-path \\
            --from "Kaiser Wilhelm II" --to "Berlin" --source der_tag

        # Show detailed connection info
        newspaper-explorer analyze entities find-path \\
            --from "Bismarck" --to "Reichstag" \\
            --show-details
    """
    from newspaper_explorer.analyze.entities.network import EntityNetworkAnalyzer

    try:
        click.echo(f"\nFinding path from '{source_entity}' to '{target_entity}'")
        click.echo(f"Source: {source}")
        if method_id:
            click.echo(f"Method ID: {method_id}")
        click.echo(f"Connection type: {connection_type}")
        click.echo(f"Max degrees: {max_degrees}")
        click.echo("")

        # Build network
        analyzer = EntityNetworkAnalyzer(source_name=source, method_id=method_id)
        analyzer.build_network(
            connection_type=connection_type,
            min_co_occurrences=min_cooccur,
        )

        # Find path
        path = analyzer.find_path(source_entity, target_entity, max_degrees=max_degrees)

        if path is None:
            click.echo(f"\n[ERROR] No path found within {max_degrees} degrees of separation")
            click.echo("\nTips:")
            click.echo("  - Check entity names (case-sensitive)")
            click.echo("  - Try increasing --max-degrees")
            click.echo("  - Try different --connection-type (date allows more connections)")
            click.echo("  - Use 'network' command to see available entities")
            return

        # Show path
        click.echo("=" * 80)
        click.echo(f"[OK] FOUND PATH: {path.degrees} DEGREES OF SEPARATION")
        click.echo("=" * 80)
        click.echo("")

        for i, entity in enumerate(path.path):
            if i == 0:
                click.echo(f"START: {entity}")
            elif i == len(path.path) - 1:
                click.echo(f"  └─► {entity} (TARGET)")
            else:
                click.echo(f"  └─► {entity}")

            # Show connection details
            if show_details and i < len(path.connections):
                conn = path.connections[i]
                click.echo(f"      Co-occurrences: {conn.co_occurrences}")
                click.echo(
                    f"      Dates: {', '.join(conn.dates[:3])}"
                    + (f" (and {len(conn.dates) - 3} more)" if len(conn.dates) > 3 else "")
                )
                if conn.pages and conn.pages[0] != "unknown":
                    click.echo(f"      Pages: {len(conn.pages)} pages")
                click.echo("")

        click.echo("=" * 80)
        click.echo(f"\nConnection summary:")
        click.echo(f"  Total path length: {path.degrees} step(s)")
        click.echo(f"  Total co-occurrences: {sum(c.co_occurrences for c in path.connections)}")

        unique_dates = set()
        for conn in path.connections:
            unique_dates.update(conn.dates)
        click.echo(f"  Unique dates: {len(unique_dates)}")
        click.echo("")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@entities_group.command(name="entity-connections")
@click.option("--source", type=str, default="der_tag", help="Source name")
@click.option("--entity", type=str, required=True, help="Entity name to explore")
@click.option(
    "--method-id",
    type=str,
    default=None,
    help="Entity extraction method ID",
)
@click.option(
    "--connection-type",
    type=click.Choice(["page", "issue", "date"], case_sensitive=False),
    default="page",
    help="How to define connections",
)
@click.option(
    "--min-cooccur",
    type=int,
    default=2,
    help="Minimum co-occurrences to show",
)
@click.option(
    "--top-n",
    type=int,
    default=30,
    help="Number of connections to show",
)
def entity_connections(
    source: str,
    entity: str,
    method_id: Optional[str],
    connection_type: str,
    min_cooccur: int,
    top_n: int,
):
    """
    Show all connections for a specific entity.

    Lists all entities that co-appear with the given entity, sorted by
    number of co-occurrences.

    Example:
        # See what entities appear with "Kaiser Wilhelm II"
        newspaper-explorer analyze entities entity-connections \\
            --entity "Kaiser Wilhelm II" --top-n 50

        # Find frequently co-occurring organizations
        newspaper-explorer analyze entities entity-connections \\
            --entity "Reichstag" --min-cooccur 5
    """
    from newspaper_explorer.analyze.entities.network import EntityNetworkAnalyzer

    try:
        click.echo(f"\nFinding connections for: '{entity}'")
        click.echo(f"Source: {source}")
        click.echo(f"Connection type: {connection_type}")
        click.echo("")

        # Build network
        analyzer = EntityNetworkAnalyzer(source_name=source, method_id=method_id)
        analyzer.build_network(
            connection_type=connection_type,
            min_co_occurrences=1,  # Build full network
        )

        # Get connections
        connections = analyzer.get_entity_connections(
            entity,
            min_co_occurrences=min_cooccur,
            sort_by="co_occurrences",
        )

        if not connections:
            click.echo(f"[ERROR] Entity not found or no connections: '{entity}'")
            click.echo("\nTips:")
            click.echo("  - Check spelling (case-sensitive)")
            click.echo("  - Try lowering --min-cooccur")
            click.echo("  - Use 'network' command to see top entities")
            return

        # Show connections
        connections_to_show = connections[:top_n]

        click.echo("=" * 80)
        click.echo(f"CONNECTIONS FOR: {entity}")
        click.echo("=" * 80)
        click.echo(f"Total connections: {len(connections)}")
        click.echo(f"Showing top {len(connections_to_show)}")
        click.echo("")
        click.echo(f"{'Connected Entity':<40} {'Co-occur':<12} {'Dates':<12} {'Pages':<12}")
        click.echo("-" * 80)

        for conn in connections_to_show:
            other_entity = conn.entity2 if conn.entity1 == entity else conn.entity1
            click.echo(
                f"{other_entity:<40} {conn.co_occurrences:<12} "
                f"{len(conn.dates):<12} {len(conn.pages):<12}"
            )

        if len(connections) > top_n:
            click.echo(f"\n... and {len(connections) - top_n} more connections")

        click.echo("\n" + "=" * 80)
        click.echo("\nUse 'find-path' to find paths between this entity and others.")
        click.echo("")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()
