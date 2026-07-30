"""Tests for data CLI commands."""

from click.testing import CliRunner
import pytest

from newspaper_explorer.cli.data.commands import data


class TestDataCommands:
    """Test data CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_data_group_help(self, runner):
        """Test that data group shows help."""
        result = runner.invoke(data, ["--help"])
        assert result.exit_code == 0
        assert "Manage newspaper data" in result.output
        assert "Commands:" in result.output

    # --- Flat commands ---

    def test_list_sources(self, runner):
        """Test list-sources command."""
        result = runner.invoke(data, ["list-sources"])
        assert result.exit_code == 0
        assert "Available Data Sources" in result.output
        assert "der_tag" in result.output

    def test_info_requires_source(self, runner):
        """Test that info command requires --source."""
        result = runner.invoke(data, ["info"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_info_with_source(self, runner):
        """Test info command with valid source."""
        result = runner.invoke(data, ["info", "--source", "der_tag"])
        assert result.exit_code == 0
        assert "SOURCE INFORMATION" in result.output
        assert "Der Tag" in result.output
        assert "DOWNLOAD" in result.output

    def test_info_with_invalid_source(self, runner):
        """Test info command with invalid source."""
        result = runner.invoke(data, ["info", "--source", "nonexistent"])
        assert result.exit_code != 0

    def test_preprocess_help(self, runner):
        """Test preprocess command help."""
        result = runner.invoke(data, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "Preprocess text data" in result.output
        assert "--source" in result.output
        assert "--steps" in result.output
        assert "normalize" in result.output

    def test_preprocess_requires_source_and_steps(self, runner):
        """Test that preprocess requires --source and --steps."""
        result = runner.invoke(data, ["preprocess"])
        assert result.exit_code != 0

    def test_list_pipelines(self, runner):
        """Test list-pipelines command."""
        result = runner.invoke(data, ["list-pipelines"])
        assert result.exit_code == 0
        assert "PIPELINE" in result.output.upper()

    # --- Text group commands ---

    def test_text_group_help(self, runner):
        """Test that text group shows help."""
        result = runner.invoke(data, ["text", "--help"])
        assert result.exit_code == 0
        assert "Text data pipeline" in result.output

    def test_text_parse_help(self, runner):
        """Test parse command help."""
        result = runner.invoke(data, ["text", "parse", "--help"])
        assert result.exit_code == 0
        assert "Parse XML files" in result.output
        assert "--source" in result.output

    def test_text_parse_requires_source(self, runner):
        """Test that parse requires --source."""
        result = runner.invoke(data, ["text", "parse"])
        assert result.exit_code != 0

    def test_text_aggregate_help(self, runner):
        """Test aggregate command help."""
        result = runner.invoke(data, ["text", "aggregate", "--help"])
        assert result.exit_code == 0
        assert "Aggregate line-level data" in result.output
        assert "--source" in result.output
        assert "--force" in result.output

    def test_text_aggregate_requires_source(self, runner):
        """Test that aggregate requires --source."""
        result = runner.invoke(data, ["text", "aggregate"])
        assert result.exit_code != 0

    def test_text_download_help(self, runner):
        """Test download command help."""
        result = runner.invoke(data, ["text", "download", "--help"])
        assert result.exit_code == 0
        assert "Download newspaper data" in result.output

    def test_text_unpack_help(self, runner):
        """Test unpack command help."""
        result = runner.invoke(data, ["text", "unpack", "--help"])
        assert result.exit_code == 0
        assert "Extract" in result.output or "Unpack" in result.output
        assert "--source" in result.output

    def test_text_verify_help(self, runner):
        """Test verify command help."""
        result = runner.invoke(data, ["text", "verify", "--help"])
        assert result.exit_code == 0
        assert "Verify" in result.output
        assert "checksum" in result.output.lower()

    # --- Images group commands ---

    def test_images_group_help(self, runner):
        """Test that images group shows help."""
        result = runner.invoke(data, ["images", "--help"])
        assert result.exit_code == 0
        assert "Image" in result.output

    def test_images_download_help(self, runner):
        """Test images download command help."""
        result = runner.invoke(data, ["images", "download", "--help"])
        assert result.exit_code == 0
        assert "Download high-resolution" in result.output
        assert "--source" in result.output

    def test_images_index_help(self, runner):
        """Test images index command help."""
        result = runner.invoke(data, ["images", "index", "--help"])
        assert result.exit_code == 0
        assert "image index" in result.output.lower()

    # --- Validation group commands ---

    def test_validation_group_help(self, runner):
        """Test that validation group shows help."""
        result = runner.invoke(data, ["validation", "--help"])
        assert result.exit_code == 0
        assert "validation" in result.output.lower()

    def test_validation_all_help(self, runner):
        """Test validation all command help."""
        result = runner.invoke(data, ["validation", "all", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output

    # --- Command listing ---

    def test_all_top_level_commands_listed(self, runner):
        """Test that all expected top-level commands and groups are listed."""
        result = runner.invoke(data, ["--help"])
        assert result.exit_code == 0

        expected = [
            "text",
            "images",
            "validation",
            "info",
            "list-sources",
            "preprocess",
            "list-pipelines",
            "analyze-chars",
            "analyze-tokens",
            "longest-tokens",
        ]

        for cmd in expected:
            assert cmd in result.output, f"Command '{cmd}' not found in help output"

    def test_all_text_commands_listed(self, runner):
        """Test that all text subcommands are listed."""
        result = runner.invoke(data, ["text", "--help"])
        assert result.exit_code == 0

        expected = ["download", "unpack", "verify", "parse", "aggregate"]
        for cmd in expected:
            assert cmd in result.output, f"Command '{cmd}' not found in text help output"


class TestCommandConsistency:
    """Test that commands follow consistent patterns."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_text_commands_requiring_source(self, runner):
        """Test that text commands all require --source."""
        source_commands = [
            ["text", "parse"],
            ["text", "aggregate"],
            ["text", "unpack"],
        ]

        for cmd_parts in source_commands:
            result = runner.invoke(data, cmd_parts)
            cmd_str = " ".join(cmd_parts)
            assert result.exit_code != 0, f"Command '{cmd_str}' should require --source"
            assert "source" in result.output.lower(), f"Command '{cmd_str}' error should mention source"

    def test_all_commands_have_help(self, runner):
        """Test that all top-level commands have --help."""
        result = runner.invoke(data, ["--help"])
        commands = []

        # Extract command names from help output
        in_commands = False
        for line in result.output.split("\n"):
            if "Commands:" in line:
                in_commands = True
                continue
            if in_commands and line.strip():
                parts = line.strip().split()
                if parts:
                    commands.append(parts[0])

        # Test each command has help
        for cmd in commands:
            result = runner.invoke(data, [cmd, "--help"])
            assert result.exit_code == 0, f"Command '{cmd}' should have --help"
            assert len(result.output) > 0, f"Command '{cmd}' help should not be empty"


class TestDeprecatedCommands:
    """Test that old command names are properly deprecated."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_old_load_command_not_available(self, runner):
        """Test that 'load' command is no longer available (now 'text parse')."""
        result = runner.invoke(data, ["load", "--help"])
        assert result.exit_code != 0

    def test_old_extract_command_not_available(self, runner):
        """Test that 'extract' command is no longer available (now 'text unpack')."""
        result = runner.invoke(data, ["extract", "--help"])
        assert result.exit_code != 0

    def test_old_status_command_not_available(self, runner):
        """Test that 'status' command is no longer available (now 'info')."""
        result = runner.invoke(data, ["status", "--help"])
        assert result.exit_code != 0

    def test_old_flat_parse_not_available(self, runner):
        """Test that 'parse' is not at top level (now under 'text')."""
        result = runner.invoke(data, ["parse", "--help"])
        assert result.exit_code != 0

    def test_old_flat_download_not_available(self, runner):
        """Test that 'download' is not at top level (now under 'text')."""
        result = runner.invoke(data, ["download", "--help"])
        assert result.exit_code != 0
