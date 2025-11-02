"""Validation commands for the data CLI."""

from pathlib import Path

import click


def register_validation_commands(data_group):
    """Register all validation-related commands to the data group."""

    @data_group.command("validate-alto-mets")
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--save-report",
        type=click.Path(),
        help="Save orphaned/unlisted ALTO files to a report file",
    )
    def validate_alto_mets(source, save_report):
        """
        Validate ALTO-METS relationships for a source.

        Checks that each ALTO file:
        1. Has a parent METS file
        2. Is properly referenced in that METS file

        This helps identify orphaned ALTO files or data integrity issues.

        \b
        Examples:
          newspaper-explorer data validate-alto-mets --source der_tag
          newspaper-explorer data validate-alto-mets --source der_tag --save-report issues.txt
        """
        from newspaper_explorer.data.utils.validation import validate_alto_mets_relationship

        try:
            click.echo(f"Validating ALTO-METS relationships for source: {source}\n")

            result = validate_alto_mets_relationship(source)

            # Display summary
            click.echo("\n" + "=" * 60)
            click.echo("ALTO-METS RELATIONSHIP VALIDATION")
            click.echo("=" * 60)
            click.echo(f"Total ALTO files:              {result['total_alto_files']:,}")
            click.echo(f"ALTO with valid METS:          {result['alto_with_mets']:,}")
            click.echo(f"ALTO without parent METS:      {result['alto_without_mets']:,}")
            click.echo(f"ALTO not listed in METS:       {result['alto_not_in_mets']:,}")
            click.echo("=" * 60)

            # Show issues if any
            if result["alto_without_mets"] > 0 or result["alto_not_in_mets"] > 0:
                click.echo("\n⚠ Issues found:")

                if result["alto_without_mets"] > 0:
                    click.echo(
                        f"\n  {result['alto_without_mets']} ALTO file(s) without parent METS:"
                    )
                    for path in result["orphaned_alto_list"][:10]:  # Show first 10
                        click.echo(f"    - {path}")
                    if len(result["orphaned_alto_list"]) > 10:
                        remaining = len(result["orphaned_alto_list"]) - 10
                        click.echo(f"    ... and {remaining} more")

                if result["alto_not_in_mets"] > 0:
                    click.echo(
                        f"\n  {result['alto_not_in_mets']} ALTO file(s) not referenced in METS:"
                    )
                    for path in result["unlisted_alto_list"][:10]:  # Show first 10
                        click.echo(f"    - {path}")
                    if len(result["unlisted_alto_list"]) > 10:
                        remaining = len(result["unlisted_alto_list"]) - 10
                        click.echo(f"    ... and {remaining} more")

                # Save report if requested
                if save_report:
                    report_path = Path(save_report)
                    with open(report_path, "w") as f:
                        f.write("ALTO-METS RELATIONSHIP VALIDATION REPORT\n")
                        f.write("=" * 60 + "\n\n")

                        if result["orphaned_alto_list"]:
                            f.write(
                                f"ALTO files without parent METS ({len(result['orphaned_alto_list'])}):\n"
                            )
                            for path in result["orphaned_alto_list"]:
                                f.write(f"  {path}\n")
                            f.write("\n")

                        if result["unlisted_alto_list"]:
                            f.write(
                                f"ALTO files not referenced in METS ({len(result['unlisted_alto_list'])}):\n"
                            )
                            for path in result["unlisted_alto_list"]:
                                f.write(f"  {path}\n")

                    click.echo(f"\n✓ Report saved to: {report_path}")
            else:
                click.echo("\n✓ All ALTO files have valid METS relationships!")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise click.Abort()

    return validate_alto_mets
