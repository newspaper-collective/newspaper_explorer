"""Tests for data CLI commands."""

import pytest
from click.testing import CliRunner

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

    def test_list_sources(self, runner):
        """Test list-sources command."""
        result = runner.invoke(data, ["list-sources"])
        assert result.exit_code == 0
        assert "Available Data Sources" in result.output
        assert "der_tag" in result.output

    def test_sources_alias(self, runner):
        """Test that 'sources' works as alias for 'list-sources'."""
        result = runner.invoke(data, ["sources"])
        assert result.exit_code == 0
        assert "Available Data Sources" in result.output

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
        assert "DOWNLOAD & EXTRACTION STATUS" in result.output

    def test_info_with_invalid_source(self, runner):
        """Test info command with invalid source."""
        result = runner.invoke(data, ["info", "--source", "nonexistent"])
        assert result.exit_code != 0

    def test_parse_help(self, runner):
        """Test parse command help."""
        result = runner.invoke(data, ["parse", "--help"])
        assert result.exit_code == 0
        assert "Parse XML files" in result.output
        assert "--source" in result.output
        assert "--resume" in result.output

    def test_parse_requires_source(self, runner):
        """Test that parse requires --source."""
        result = runner.invoke(data, ["parse"])
        assert result.exit_code != 0

    def test_aggregate_help(self, runner):
        """Test aggregate command help."""
        result = runner.invoke(data, ["aggregate", "--help"])
        assert result.exit_code == 0
        assert "Aggregate line-level data" in result.output
        assert "--source" in result.output
        assert "--force" in result.output

    def test_aggregate_requires_source(self, runner):
        """Test that aggregate requires --source."""
        result = runner.invoke(data, ["aggregate"])
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

    def test_download_help(self, runner):
        """Test download command help."""
        result = runner.invoke(data, ["download", "--help"])
        assert result.exit_code == 0
        assert "Download newspaper data" in result.output

    def test_unpack_help(self, runner):
        """Test unpack command help."""
        result = runner.invoke(data, ["unpack", "--help"])
        assert result.exit_code == 0
        assert "Unpack" in result.output or "extract" in result.output
        assert "--source" in result.output

    def test_verify_help(self, runner):
        """Test verify command help."""
        result = runner.invoke(data, ["verify", "--help"])
        assert result.exit_code == 0
        assert "Verify" in result.output
        assert "checksum" in result.output.lower()

    def test_download_images_help(self, runner):
        """Test download-images command help."""
        result = runner.invoke(data, ["download-images", "--help"])
        assert result.exit_code == 0
        assert "Download high-resolution" in result.output
        assert "--source" in result.output

    def test_find_empty_help(self, runner):
        """Test find-empty command help."""
        result = runner.invoke(data, ["find-empty", "--help"])
        assert result.exit_code == 0
        assert "Find XML files without" in result.output
        assert "--source" in result.output

    def test_find_empty_requires_source(self, runner):
        """Test that find-empty requires --source."""
        result = runner.invoke(data, ["find-empty"])
        assert result.exit_code != 0

    def test_all_commands_listed(self, runner):
        """Test that all expected commands are listed."""
        result = runner.invoke(data, ["--help"])
        assert result.exit_code == 0

        expected_commands = [
            "aggregate",
            "download",
            "download-images",
            "find-empty",
            "info",
            "list-sources",
            "parse",
            "preprocess",
            "unpack",
            "verify",
        ]

        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in help output"


class TestCommandConsistency:
    """Test that commands follow consistent patterns."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_commands_requiring_source(self, runner):
        """Test that source-based commands all require --source."""
        source_commands = [
            "info",
            "parse",
            "aggregate",
            "preprocess",
            "download-images",
            "find-empty",
            "unpack",
        ]

        for cmd in source_commands:
            result = runner.invoke(data, [cmd])
            assert result.exit_code != 0, f"Command '{cmd}' should require --source"
            assert "source" in result.output.lower(), f"Command '{cmd}' error should mention source"

    def test_all_commands_have_help(self, runner):
        """Test that all commands have --help."""
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
        """Test that 'load' command is no longer available (now 'parse')."""
        result = runner.invoke(data, ["load", "--help"])
        assert result.exit_code != 0

    def test_old_extract_command_not_available(self, runner):
        """Test that 'extract' command is no longer available (now 'unpack')."""
        result = runner.invoke(data, ["extract", "--help"])
        assert result.exit_code != 0

    def test_old_status_command_not_available(self, runner):
        """Test that 'status' command is no longer available (now 'info')."""
        result = runner.invoke(data, ["status", "--help"])
        assert result.exit_code != 0

    def test_old_load_status_command_not_available(self, runner):
        """Test that 'load-status' command is no longer available (now 'info')."""
        result = runner.invoke(data, ["load-status", "--help"])
        assert result.exit_code != 0
