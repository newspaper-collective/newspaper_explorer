"""
Universal results loader utility for analysis results.

Provides consistent interface for loading analysis results with metadata
from the results directory structure: results/{source}/{analysis_type}/{run_id}/
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import polars as pl

from newspaper_explorer.config.base import get_config


class AnalysisResult:
    """Container for analysis results with metadata"""

    def __init__(
        self,
        source: str,
        analysis_type: str,
        run_id: str,
        metadata: dict,
        parquet_path: Path,
    ):
        self.source = source
        self.analysis_type = analysis_type
        self.run_id = run_id
        self.metadata = metadata
        self.parquet_path = parquet_path
        self._df: Optional[pl.DataFrame] = None

    @property
    def display_name(self) -> str:
        """Generate display name from metadata"""
        method = self.metadata.get("method_type", "unknown")
        model = self.metadata.get("model_name", "unknown")
        created = self.metadata.get("created_at", "")

        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = created[:19]  # Fallback to ISO format
        else:
            date_str = "unknown date"

        return f"{method} ({model}) - {date_str}"

    @property
    def df(self) -> pl.DataFrame:
        """Lazy load DataFrame"""
        if self._df is None:
            self._df = pl.read_parquet(self.parquet_path)
        return self._df

    def reload(self):
        """Force reload of DataFrame"""
        self._df = pl.read_parquet(self.parquet_path)

    @property
    def row_count(self) -> int:
        """Get row count from metadata or DataFrame"""
        if "output_data" in self.metadata and "row_count" in self.metadata["output_data"]:
            return self.metadata["output_data"]["row_count"]
        return len(self.df)

    @property
    def created_at(self) -> Optional[datetime]:
        """Parse created_at timestamp"""
        created = self.metadata.get("created_at")
        if created:
            try:
                return datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get analysis duration in seconds"""
        return self.metadata.get("duration_seconds")

    @property
    def parameters(self) -> dict:
        """Get analysis parameters"""
        return self.metadata.get("parameters", {})

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "source": self.source,
            "analysis_type": self.analysis_type,
            "run_id": self.run_id,
            "display_name": self.display_name,
            "row_count": self.row_count,
            "created_at": self.metadata.get("created_at"),
            "duration_seconds": self.duration_seconds,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }


class ResultsLoader:
    """Universal loader for analysis results"""

    def __init__(self) -> None:
        self.config = get_config()
        self.results_dir = Path(self.config.results_dir)

    def list_sources(self) -> list[str]:
        """List all sources with results"""
        if not self.results_dir.exists():
            return []
        return [d.name for d in self.results_dir.iterdir() if d.is_dir()]

    def list_analysis_types(self, source: str) -> list[str]:
        """List all analysis types available for a source"""
        source_dir = self.results_dir / source
        if not source_dir.exists():
            return []
        return [d.name for d in source_dir.iterdir() if d.is_dir()]

    def list_runs(self, source: str, analysis_type: str) -> list[tuple[str, str]]:
        """
        List all runs for a source and analysis type.

        Returns:
            List of tuples (run_id, display_name)
        """
        # Special handling for text analysis type
        if analysis_type == "text":
            text_dir = Path(self.config.data_dir) / "raw" / source / "text"
            if not text_dir.exists():
                return []

            runs = []
            for file_path in text_dir.glob("*.parquet"):
                run_id = file_path.name
                display_name = file_path.stem.replace("_", " ").title()
                runs.append((run_id, display_name))

            # Sort by name
            runs.sort(key=lambda x: x[0])
            return runs

        analysis_dir = self.results_dir / source / analysis_type
        if not analysis_dir.exists():
            return []

        runs = []
        for run_dir in analysis_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metadata_path = run_dir / f"{analysis_type}.json"
            if not metadata_path.exists():
                continue

            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)

                result = AnalysisResult(
                    source=source,
                    analysis_type=analysis_type,
                    run_id=run_dir.name,
                    metadata=metadata,
                    parquet_path=run_dir / f"{analysis_type}.parquet",
                )
                runs.append((run_dir.name, result.display_name))
            except Exception as e:
                print(f"Error loading metadata for {run_dir}: {e}")
                continue

        # Sort by creation date (newest first)
        runs.sort(key=lambda x: x[0], reverse=True)
        return runs

    def load_result(
        self, source: str, analysis_type: str, run_id: Optional[str] = None
    ) -> Optional[AnalysisResult]:
        """
        Load a specific analysis result.

        Args:
            source: Source name
            analysis_type: Analysis type (entities, emotions, etc.)
            run_id: Optional run ID. If None, loads the most recent run.

        Returns:
            AnalysisResult or None if not found
        """
        # Special handling for text analysis type
        if analysis_type == "text":
            text_dir = Path(self.config.data_dir) / "raw" / source / "text"

            if run_id is None:
                # Default to the main lines file if available
                default_file = text_dir / f"{source}_lines.parquet"
                if default_file.exists():
                    run_id = default_file.name
                else:
                    runs = self.list_runs(source, analysis_type)
                    if not runs:
                        return None
                    run_id = runs[0][0]

            parquet_path = text_dir / run_id
            if not parquet_path.exists():
                return None

            # Create synthetic metadata
            try:
                stats = parquet_path.stat()
                created_at = datetime.fromtimestamp(stats.st_mtime).isoformat()

                # Get row count efficiently using DuckDB
                row_count = 0
                try:
                    import duckdb

                    con = duckdb.connect()
                    result = con.execute(
                        f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
                    ).fetchone()
                    if result:
                        row_count = result[0]
                except Exception as e:
                    print(f"Error counting rows for {parquet_path}: {e}")

                metadata = {
                    "created_at": created_at,
                    "method_type": "Text Import",
                    "model_name": "Raw/Preprocessed",
                    "parameters": {},
                    "output_data": {"row_count": row_count},
                }

                return AnalysisResult(
                    source=source,
                    analysis_type=analysis_type,
                    run_id=run_id,
                    metadata=metadata,
                    parquet_path=parquet_path,
                )
            except Exception as e:
                print(f"Error loading text result: {e}")
                return None

        analysis_dir = self.results_dir / source / analysis_type
        if not analysis_dir.exists():
            return None

        # If no run_id specified, get the most recent
        if run_id is None:
            runs = self.list_runs(source, analysis_type)
            if not runs:
                return None
            run_id = runs[0][0]

        run_dir = analysis_dir / run_id
        if not run_dir.exists():
            return None

        metadata_path = run_dir / f"{analysis_type}.json"
        parquet_path = run_dir / f"{analysis_type}.parquet"

        if not metadata_path.exists() or not parquet_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            return AnalysisResult(
                source=source,
                analysis_type=analysis_type,
                run_id=run_id,
                metadata=metadata,
                parquet_path=parquet_path,
            )
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
            print(f"Error loading result: {e}")
            return None

    def get_latest_result(self, source: str, analysis_type: str) -> Optional[AnalysisResult]:
        """Get the most recent result for a source and analysis type"""
        return self.load_result(source, analysis_type, run_id=None)

    def check_availability(self, source: str, analysis_type: str) -> bool:
        """Check if any results exist for source and analysis type"""
        runs = self.list_runs(source, analysis_type)
        return len(runs) > 0
