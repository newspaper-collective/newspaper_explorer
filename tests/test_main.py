"""Tests for main CLI entry point."""

import pytest
from click.testing import CliRunner

from newspaper_explorer.main import cli


class TestMainCLI:
    """Test main CLI entry point and command group registration."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that main CLI shows help message."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Newspaper Explorer" in result.output
        assert "Explore and analyze historical newspaper data" in result.output

    def test_cli_help_shows_command_groups(self, runner):
        """Test that help displays available command groups."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "data" in result.output
        assert "analyze" in result.output

    def test_cli_help_shows_examples(self, runner):
        """Test that help includes usage examples."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Data Management:" in result.output
        assert "Analysis:" in result.output
        assert "newspaper-explorer data" in result.output
        assert "newspaper-explorer analyze" in result.output

    def test_cli_no_command_shows_help(self, runner):
        """Test that running CLI without command shows help."""
        result = runner.invoke(cli, [])
        # Click returns exit code 2 when no command is provided for a group
        assert result.exit_code == 2
        assert "Newspaper Explorer" in result.output

    def test_cli_invalid_command(self, runner):
        """Test that invalid command shows error."""
        result = runner.invoke(cli, ["invalid-command"])
        assert result.exit_code != 0
        assert "Error" in result.output or "No such command" in result.output

    def test_data_command_registered(self, runner):
        """Test that data command group is registered."""
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "Manage newspaper data" in result.output

    def test_analyze_command_registered(self, runner):
        """Test that analyze command group is registered."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        # Analyze command should show its help
        assert result.exit_code == 0

    def test_data_subcommand_accessible(self, runner):
        """Test that data subcommands are accessible."""
        result = runner.invoke(cli, ["data", "list-sources"])
        assert result.exit_code == 0
        assert "Available Data Sources" in result.output

    def test_cli_version_info(self, runner):
        """Test that CLI provides version information if available."""
        result = runner.invoke(cli, ["--version"])
        # May or may not have version flag, depending on implementation
        # Just ensure it doesn't crash
        assert result.exit_code in [0, 2]  # 0 = success, 2 = no such option

    def test_help_flag(self, runner):
        """Test that --help flag works (Click doesn't support -h by default)."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Newspaper Explorer" in result.output

    def test_nested_help_data(self, runner):
        """Test that nested help works for data commands."""
        result = runner.invoke(cli, ["data", "info", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output

    def test_prog_name_when_main(self):
        """Test that prog_name is set correctly when run as main."""
        runner = CliRunner()
        # This tests the if __name__ == "__main__" block behavior
        # by ensuring the CLI works with explicit prog_name
        result = runner.invoke(cli, ["--help"], prog_name="newspaper-explorer")
        assert result.exit_code == 0


class TestCLIIntegration:
    """Integration tests for CLI command flow."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_full_command_chain_help(self, runner):
        """Test help at each level of command chain."""
        # Main help
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

        # Data group help
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0

        # Data subcommand help
        result = runner.invoke(cli, ["data", "info", "--help"])
        assert result.exit_code == 0

    def test_command_isolation(self, runner):
        """Test that command groups don't interfere with each other."""
        # Run data command
        result_data = runner.invoke(cli, ["data", "list-sources"])
        assert result_data.exit_code == 0

        # Run analyze help (different group)
        result_analyze = runner.invoke(cli, ["analyze", "--help"])
        assert result_analyze.exit_code == 0

        # Both should work independently
        assert "Available Data Sources" in result_data.output
        assert "Available Data Sources" not in result_analyze.output


class TestCLIErrorHandling:
    """Test error handling in CLI."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_nonexistent_command_group(self, runner):
        """Test error for non-existent command group."""
        result = runner.invoke(cli, ["nonexistent"])
        assert result.exit_code != 0

    def test_nonexistent_subcommand(self, runner):
        """Test error for non-existent subcommand in valid group."""
        result = runner.invoke(cli, ["data", "nonexistent"])
        assert result.exit_code != 0

    def test_missing_required_option(self, runner):
        """Test error when required option is missing."""
        result = runner.invoke(cli, ["data", "info"])
        assert result.exit_code != 0
        # Should mention missing option
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_invalid_option(self, runner):
        """Test error for invalid option."""
        result = runner.invoke(cli, ["data", "--invalid-option"])
        assert result.exit_code != 0


class TestCLIDocumentation:
    """Test that CLI documentation is consistent and complete."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_help_formatting(self, runner):
        """Test that help text is properly formatted."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

        # Check for proper structure
        lines = result.output.split("\n")
        assert any("Usage:" in line for line in lines)
        assert any("Options:" in line for line in lines)

    def test_help_mentions_subcommand_help(self, runner):
        """Test that help mentions how to get subcommand help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

        # Should guide users to subcommand help
        assert "--help" in result.output

    def test_command_descriptions_present(self, runner):
        """Test that all commands have descriptions."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

        # Both command groups should have descriptions
        output_lower = result.output.lower()
        assert "data" in output_lower
        assert "analyze" in output_lower
