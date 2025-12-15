"""
CLI commands for entity extraction.

Provides commands for extracting named entities using GLiNER or LLM methods.
"""

import json
import logging
from typing import Optional

import click

from newspaper_explorer.analyze.entities.gliner import GLINER_MODELS, GLiNEREntityExtractor
from newspaper_explorer.analyze.entities.gliner2 import GLiNER2EntityExtractor
from newspaper_explorer.analyze.entities.llm import LLMEntityExtractor
from newspaper_explorer.analyze.entities.network import EntityNetworkAnalyzer
from newspaper_explorer.analyze.entities.utils import (
    compare_existing_results,
    list_entity_results,
    print_comparison_report,
)
from newspaper_explorer.cli.utils import errors, output, paths
from newspaper_explorer.cli.utils.options import (
    batch_size_option,
    input_file_option,
    limit_option,
    max_length_option,
    max_tokens_option,
    min_length_option,
    model_option,
    num_gpus_option,
    source_option,
    temperature_option,
    text_column_option,
    threshold_option,
)
from newspaper_explorer.config.base import get_config

# Display constants
MAX_MODEL_NAME_LENGTH = 20


@click.group(name="entities")
def entities_group() -> None:
    """Extract named entities from newspaper text."""
    pass


@entities_group.command(name="list-models")
def list_models() -> None:
    """List available GLiNER models."""

    output.header("Available GLiNER Models")

    for short_name, info in GLINER_MODELS.items():
        default_marker = " (default)" if info.get("default") else ""
        output.section(f"{short_name}{default_marker}")
        output.key_value("Model ID", info["model_id"])
        output.key_value("Size", info["size"])
        output.key_value("Max tokens", info["max_tokens"])
        output.key_value("Description", info["description"])
        click.echo()

    output.section("Usage Examples")
    output.code_block(
        [
            "# Use short name with labels",
            "newspaper-explorer analyze entities gliner --source der_tag \\",
            "  --labels 'Person,Organisation,Ort' --model xxl-v2.5",
            "",
            "# Use full Hugging Face model ID",
            "newspaper-explorer analyze entities gliner --source der_tag \\",
            "  --labels 'Person,Organisation,Ort' --model 'urchade/gliner_multi-v2.1'",
        ]
    )
    click.echo()


@entities_group.command(name="gliner")
@source_option()
@click.option("--labels", type=str, required=True, help="Comma-separated entity labels to extract")
@input_file_option(help_text="Custom input parquet file (default: textblocks or lines)")
@text_column_option()
@click.option(
    "--id-column",
    type=str,
    help="Name of ID column (auto-detected: text_block_id or line_id)",
)
@model_option(
    default="multi-v2.1",
    help_text="GLiNER model (multi-v2.1, xxl-v2.5, small-v2.1, large-v2.1, or full HF ID)",
)
@threshold_option(default=0.2)
@batch_size_option(default=32)
@min_length_option(default=10)
@max_length_option(default=1000, help_text="Maximum text length (~1000 chars = ~200 tokens)")
@limit_option()
@num_gpus_option(default=1)
def gliner(
    source: str,
    labels: str,
    input_file: Optional[str],
    text_column: str,
    id_column: Optional[str],
    model: str,
    threshold: float,
    batch_size: int,
    min_length: int,
    max_length: int,
    limit: Optional[int],
    num_gpus: int,
) -> None:
    """Extract named entities using GLiNER model (fast, local, offline)."""

    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    output.header("GLiNER Entity Extraction")

    label_list = [label.strip() for label in labels.split(",")]

    output.key_value("Source", source)
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)
    output.key_value("Model", model)
    output.key_value("Labels", ", ".join(label_list))
    output.key_value("Threshold", str(threshold))
    output.key_value("Batch size", batch_size)
    if limit:
        output.key_value("Limit", f"{limit:,} rows")
    output.key_value("GPUs", num_gpus)
    click.echo()

    try:
        # Resolve input path and ID column
        input_path, id_col = paths.resolve_input_path(source, input_file, id_column)

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

        click.echo()
        output.success("Entity extraction complete!")
        output.info(f"Method ID: {results['metadata'].analysis_id}")
        output.info(f"Entities extracted: {len(results['results_df']):,}")
        output.info(f"Results saved to: {results['output_dir']}")
        click.echo()

    except FileNotFoundError as e:
        click.echo()
        errors.handle_validation_error(
            f"File not found: {e}",
            issues=[
                f"Run parsing first: newspaper-explorer data parse --source {source}",
                "Specify custom input: --input-file path/to/file.parquet",
            ],
        )
    except (RuntimeError, ValueError, OSError) as e:
        click.echo()
        errors.handle_error(e, show_traceback=True)


@entities_group.command(name="gliner2")
@source_option()
@click.option("--labels", type=str, required=True, help="Comma-separated entity labels to extract")
@click.option(
    "--label-descriptions",
    type=str,
    help='JSON dict with label descriptions for better accuracy (e.g., \'{"Person":"Names of people"}\')',
)
@input_file_option(help_text="Custom input parquet file (default: textblocks or lines)")
@text_column_option()
@click.option(
    "--id-column",
    type=str,
    help="Name of ID column (auto-detected: text_block_id or line_id)",
)
@model_option(
    default="fastino/gliner2-base-0207",
    help_text="GLiNER2 model (base=205M, large=340M)",
)
@threshold_option(default=0.5)
@batch_size_option(default=32)
@min_length_option(default=100)
@max_length_option(default=2500, help_text="Maximum text length before chunking")
@limit_option()
def gliner2(
    source: str,
    labels: str,
    label_descriptions: Optional[str],
    input_file: Optional[str],
    text_column: str,
    id_column: Optional[str],
    model: str,
    threshold: float,
    batch_size: int,
    min_length: int,
    max_length: int,
    limit: Optional[int],
) -> None:
    """Extract named entities using GLiNER2 (optimized CPU inference, 205M params)."""

    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    output.header("GLiNER2 Entity Extraction")

    # Parse labels
    label_list = [label.strip() for label in labels.split(",")]

    # Parse label descriptions if provided (GLiNER2-specific)
    if label_descriptions:
        try:
            parsed_dict = json.loads(label_descriptions)
            if not isinstance(parsed_dict, dict):
                errors.handle_validation_error(
                    "Invalid label descriptions format",
                    issues=["--label-descriptions must be a JSON dict"],
                )
                raise click.Abort()
            labels_dict = parsed_dict
        except json.JSONDecodeError as e:
            errors.handle_validation_error(
                f"Error parsing JSON: {e}",
                issues=["Check your JSON syntax in --label-descriptions"],
            )
            raise click.Abort() from None
    else:
        # Use German descriptions by default for better accuracy
        labels_dict = {
            label: f"{label} in German historical newspaper text" for label in label_list
        }

    output.key_value("Source", source)
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)
    output.key_value("Model", f"{model} (CPU-optimized)")
    output.key_value("Labels", ", ".join(label_list))
    output.key_value("Label descriptions", "Yes" if label_descriptions else "Auto-generated")
    output.key_value("Threshold", str(threshold))
    output.key_value("Batch size", batch_size)
    if limit:
        output.key_value("Limit", f"{limit:,} rows")
    click.echo()

    try:
        # Resolve input path and ID column
        input_path, id_col = paths.resolve_input_path(source, input_file, id_column)

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

        click.echo()
        output.success("Entity extraction complete!")
        output.info(f"Method ID: {results['metadata'].analysis_id}")
        output.info(f"Entities extracted: {len(results['results_df']):,}")
        output.info(f"Results saved to: {results['output_dir']}")
        click.echo()

    except FileNotFoundError as e:
        click.echo()
        errors.handle_validation_error(
            f"File not found: {e}",
            issues=[
                f"Run parsing first: newspaper-explorer data parse --source {source}",
                "Specify custom input: --input-file path/to/file.parquet",
            ],
        )
    except (RuntimeError, ValueError, OSError) as e:
        click.echo()
        errors.handle_error(e, show_traceback=True)


@entities_group.command(name="llm")
@source_option()
@input_file_option(help_text="Custom input parquet file (default: textblocks or lines)")
@text_column_option()
@click.option(
    "--id-column",
    type=str,
    help="Name of ID column (auto-detected)",
)
@model_option(default="gpt-4o-mini", help_text="LLM model")
@temperature_option(default=0.3)
@max_tokens_option(default=2000)
@min_length_option(default=100)
@max_length_option(default=8000, help_text="Maximum text length before chunking (chars)")
@batch_size_option(default=10)
@limit_option()
def llm(
    source: str,
    input_file: Optional[str],
    text_column: str,
    id_column: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    min_length: int,
    max_length: int,
    batch_size: int,
    limit: Optional[int],
) -> None:
    """Extract named entities using LLM (high quality, requires API)."""

    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    output.header("LLM Entity Extraction")

    output.key_value("Source", source)
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)
    output.key_value("Model", model)
    output.key_value("Temperature", str(temperature))
    output.key_value("Max tokens", max_tokens)
    output.key_value("Text length", f"min={min_length}, max={max_length} chars")
    output.key_value("Batch size", batch_size)
    if limit:
        output.key_value("Limit", f"{limit:,} rows")
    click.echo()

    try:
        # Resolve input path and ID column
        input_path, id_col = paths.resolve_input_path(source, input_file, id_column)

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

        click.echo()
        output.success("Entity extraction complete!")
        output.info(f"Method ID: {results['metadata'].analysis_id}")
        output.info(f"Entities extracted: {len(results['results_df']):,}")
        output.info(f"Results saved to: {results['output_dir']}")
        click.echo()

    except FileNotFoundError as e:
        click.echo()
        errors.handle_validation_error(
            f"File not found: {e}",
            issues=[
                f"Run parsing first: newspaper-explorer data parse --source {source}",
                "Specify custom input: --input-file path/to/file.parquet",
            ],
        )
    except (RuntimeError, ValueError, OSError) as e:
        click.echo()
        errors.handle_error(e, show_traceback=True)


@entities_group.command(name="list-results")
@source_option()
def list_results(source: str) -> None:
    """
    List all available entity extraction results for a source.

    Shows method IDs, models used, entity counts, and timestamps.

    Example:
        newspaper-explorer analyze entities list-results --source der_tag
    """

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
            model = result["model"][:18] + ".." if len(result["model"]) > MAX_MODEL_NAME_LENGTH else result["model"]
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

    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort() from None


@entities_group.command(name="compare")
@source_option()
@click.option("--method-1", type=str, required=True, help="First method ID to compare")
@click.option("--method-2", type=str, required=True, help="Second method ID to compare")
@click.option(
    "--detailed/--summary",
    default=False,
    help="Show detailed comparison (default: summary only)",
)
def compare_results(source: str, method_1: str, method_2: str, *, detailed: bool) -> None:
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
        raise click.Abort() from None
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort() from None


@entities_group.command(name="network")
@source_option()
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
) -> None:
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

    try:
        output.header("Entity Network Analysis")

        output.key_value("Source", source)
        if method_id:
            output.key_value("Method ID", method_id)
        else:
            output.info("Using most recent entity extraction")
        output.key_value("Connection type", connection_type)
        output.key_value("Min co-occurrences", min_cooccur)

        # Parse entity types
        entity_type_list = None
        if entity_types:
            entity_type_list = [t.strip() for t in entity_types.split(",")]
            output.key_value("Entity types filter", ", ".join(entity_type_list))
        click.echo()

        # Build network
        analyzer = EntityNetworkAnalyzer(source_name=source, method_id=method_id)
        stats = analyzer.build_network(
            connection_type=connection_type,
            min_co_occurrences=min_cooccur,
            entity_types=entity_type_list,
        )

        # Print network stats
        output.section("Network Statistics")
        output.key_value("Nodes (entities)", f"{stats['nodes']:,}")
        output.key_value("Edges (connections)", f"{stats['edges']:,}")
        output.key_value("Avg connections/entity", f"{stats['avg_connections_per_entity']:.2f}")

        # Get comprehensive stats
        full_stats = analyzer.get_network_stats()
        output.key_value("Max degree", full_stats["max_degree"])
        output.key_value("Min degree", full_stats["min_degree"])
        output.key_value("Total co-occurrences", f"{full_stats['total_co_occurrences']:,}")

        # Show top entities
        click.echo()
        output.section(f"Top {top_n} Most Connected Entities")
        click.echo(f"{'Entity':<40} {'Connections':<15} {'Total Co-occur':<15}")
        click.echo("─" * 70)

        top_entities = analyzer.get_top_entities(n=top_n, metric="degree")
        for entity, degree in top_entities:
            total_cooccur = analyzer.entity_metadata[entity]["total_co_occurrences"]
            click.echo(f"{entity:<40} {int(degree):<15} {total_cooccur:<15}")

        click.echo()
        output.info("Use 'find-path' command to find connections between entities", muted=True)
        click.echo()

    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        click.echo()
        errors.handle_validation_error(str(e))


@entities_group.command(name="find-path")
@source_option()
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
    *,
    show_details: bool,
) -> None:
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

    try:
        output.header("Find Entity Connection Path")

        output.key_value("From", source_entity)
        output.key_value("To", target_entity)
        output.key_value("Source", source)
        if method_id:
            output.key_value("Method ID", method_id)
        output.key_value("Connection type", connection_type)
        output.key_value("Max degrees", max_degrees)
        click.echo()

        # Build network
        analyzer = EntityNetworkAnalyzer(source_name=source, method_id=method_id)
        analyzer.build_network(
            connection_type=connection_type,
            min_co_occurrences=min_cooccur,
        )

        # Find path
        path = analyzer.find_path(source_entity, target_entity, max_degrees=max_degrees)

        if path is None:
            click.echo()
            errors.handle_validation_error(
                f"No path found within {max_degrees} degrees of separation",
                issues=[
                    "Check entity names (case-sensitive)",
                    f"Try increasing --max-degrees (current: {max_degrees})",
                    "Try different --connection-type (date allows more connections)",
                    "Use 'network-stats' command to see available entities",
                ],
            )
            return

        # Show path
        output.section(f"Found Path: {path.degrees} Degrees of Separation")
        click.echo()

        max_dates_to_show = 3
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
                dates_display = ", ".join(conn.dates[:max_dates_to_show])
                if len(conn.dates) > max_dates_to_show:
                    dates_display += f" (and {len(conn.dates) - max_dates_to_show} more)"
                click.echo(f"      Dates: {dates_display}")
                if conn.pages and conn.pages[0] != "unknown":
                    click.echo(f"      Pages: {len(conn.pages)} pages")
                click.echo()

        click.echo()
        output.section("Connection Summary")
        output.key_value("Path length", f"{path.degrees} step(s)")
        output.key_value("Total co-occurrences", sum(c.co_occurrences for c in path.connections))

        unique_dates = set()
        for conn in path.connections:
            unique_dates.update(conn.dates)
        output.key_value("Unique dates", len(unique_dates))
        click.echo()

    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        click.echo()
        errors.handle_validation_error(str(e))


@entities_group.command(name="entity-connections")
@source_option()
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
) -> None:
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

    try:
        output.header("Entity Connections")

        output.key_value("Entity", entity)
        output.key_value("Source", source)
        if method_id:
            output.key_value("Method ID", method_id)
        output.key_value("Connection type", connection_type)
        output.key_value("Min co-occurrences", min_cooccur)
        click.echo()

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
            click.echo()
            errors.handle_validation_error(
                f"Entity not found or no connections: '{entity}'",
                issues=[
                    "Check spelling (case-sensitive)",
                    f"Try lowering --min-cooccur (current: {min_cooccur})",
                    "Use 'network-stats' command to see top entities",
                ],
            )
            return

        # Show connections
        connections_to_show = connections[:top_n]

        output.section(f"Connections for: {entity}")
        output.key_value("Total connections", len(connections))
        output.key_value("Showing", len(connections_to_show))
        click.echo()
        click.echo(f"{'Connected Entity':<40} {'Co-occur':<12} {'Dates':<12} {'Pages':<12}")
        click.echo("─" * 76)

        for conn in connections_to_show:
            other_entity = conn.entity2 if conn.entity1 == entity else conn.entity1
            click.echo(
                f"{other_entity:<40} {conn.co_occurrences:<12} "
                f"{len(conn.dates):<12} {len(conn.pages):<12}"
            )

        if len(connections) > top_n:
            click.echo()
            output.info(f"... and {len(connections) - top_n} more connections", muted=True)

        click.echo()
        output.info("Use 'find-path' to find paths between this entity and others", muted=True)
        click.echo()

    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        click.echo()
        errors.handle_validation_error(str(e))
