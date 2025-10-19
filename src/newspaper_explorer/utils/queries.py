"""
Query utilities using DuckDB for efficient Parquet queries.

Provides SQL interface to source data and analysis results without
loading entire files into memory.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import polars as pl

from newspaper_explorer.utils.config import get_config

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    DuckDB-based query engine for newspaper data.

    Enables SQL queries across Parquet files (source data + analysis results)
    without loading entire files into memory.

    Example:
        ```python
        from newspaper_explorer.utils.queries import QueryEngine

        with QueryEngine() as qe:
            # Find all mentions of an entity
            mentions = qe.find_entity_mentions(
                entity_name="Kaiser Wilhelm II",
                method="llm_gpt4o_mini"
            )

            # Get full text for a line
            line = qe.get_line(line_id="der_tag_1901_01_15_001")
        ```
    """

    def __init__(
        self,
        source: str = "der_tag",
        db_path: Optional[Path] = None,
        in_memory: bool = True,
    ):
        """
        Initialize query engine.

        Args:
            source: Source name (e.g., "der_tag").
            db_path: Path to persistent DuckDB file. If None and in_memory=False,
                    uses results/{source}/query_cache.duckdb.
            in_memory: Use in-memory database (faster, no persistence).
        """
        self.source = source
        config = get_config()

        # Determine database path
        if in_memory:
            self.db_path = None
            self.con = duckdb.connect()
            logger.debug("Initialized in-memory DuckDB connection")
        else:
            if db_path is None:
                db_path = config.results_dir / source / "query_cache.duckdb"
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(str(self.db_path))
            logger.debug(f"Initialized persistent DuckDB: {self.db_path}")

        # Set up paths
        self.data_dir = config.data_dir
        self.results_dir = config.results_dir
        self.source_parquet = self.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"

        # Create views for common queries
        self._create_views()

    def _create_views(self):
        """Create common views for easier querying."""
        if not self.source_parquet.exists():
            logger.warning(f"Source parquet not found: {self.source_parquet}")
            return

        # Create view for source data
        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW source_lines AS
            SELECT * FROM '{self.source_parquet}'
        """
        )
        logger.debug("Created source_lines view")

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pl.DataFrame:
        """
        Execute SQL query and return Polars DataFrame.

        Args:
            sql: SQL query string.
            params: Optional list of parameters for parameterized query.

        Returns:
            Query result as Polars DataFrame.
        """
        if params:
            result = self.con.execute(sql, params).df()
        else:
            result = self.con.execute(sql).df()

        return pl.from_pandas(result)

    def get_line(self, line_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full text and metadata for a specific line.

        Args:
            line_id: Unique line identifier.

        Returns:
            Dictionary with line data, or None if not found.
        """
        result = self.query(
            """
            SELECT *
            FROM source_lines
            WHERE line_id = ?
            """,
            params=[line_id],
        )

        if len(result) == 0:
            return None

        return result.to_dicts()[0]

    def search_text(
        self,
        query_text: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        """
        Full-text search with optional date filtering.

        Args:
            query_text: Text to search for (case-insensitive substring match).
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).
            limit: Maximum results to return.

        Returns:
            DataFrame with matching lines.
        """
        sql = """
            SELECT line_id, text, date, filename, text_block_id
            FROM source_lines
            WHERE text ILIKE ?
        """
        params = [f"%{query_text}%"]

        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)

        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)

        sql += f" LIMIT {limit}"

        return self.query(sql, params)

    def find_entity_mentions(
        self,
        entity_name: str,
        method: str = "llm_gpt4o_mini",
        entity_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Find all mentions of an entity with full text context.

        Args:
            entity_name: Entity to search for (exact match).
            method: Analysis method ID (directory name).
            entity_type: Optional filter by entity type (person, location, organization).
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            DataFrame with entity mentions and text context.
        """
        entities_path = self.results_dir / self.source / "entities" / method / "entities.parquet"

        if not entities_path.exists():
            logger.warning(f"Entity results not found: {entities_path}")
            return pl.DataFrame()

        sql = f"""
            SELECT 
                e.entity_text,
                e.entity_type,
                e.confidence,
                s.line_id,
                s.text,
                s.date,
                s.filename,
                s.text_block_id
            FROM '{entities_path}' e
            JOIN source_lines s ON e.line_id = s.line_id
            WHERE e.entity_text = ?
        """
        params = [entity_name]

        if entity_type:
            sql += " AND e.entity_type = ?"
            params.append(entity_type)

        if start_date:
            sql += " AND s.date >= ?"
            params.append(start_date)

        if end_date:
            sql += " AND s.date <= ?"
            params.append(end_date)

        sql += " ORDER BY s.date"

        return self.query(sql, params)

    def compare_entity_methods(
        self, method1: str, method2: str, entity_type: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Compare entity extraction results from two methods.

        Args:
            method1: First method ID.
            method2: Second method ID.
            entity_type: Optional filter by entity type.

        Returns:
            DataFrame with entities found by each method (showing differences).
        """
        path1 = self.results_dir / self.source / "entities" / method1 / "entities.parquet"
        path2 = self.results_dir / self.source / "entities" / method2 / "entities.parquet"

        if not path1.exists() or not path2.exists():
            logger.warning("One or both entity result files not found")
            return pl.DataFrame()

        sql = f"""
            SELECT 
                COALESCE(e1.line_id, e2.line_id) as line_id,
                e1.entity_text as method1_entity,
                e1.entity_type as method1_type,
                e2.entity_text as method2_entity,
                e2.entity_type as method2_type,
                s.text,
                s.date
            FROM '{path1}' e1
            FULL OUTER JOIN '{path2}' e2
                ON e1.line_id = e2.line_id 
                AND e1.entity_text = e2.entity_text
            JOIN source_lines s
                ON COALESCE(e1.line_id, e2.line_id) = s.line_id
            WHERE e1.entity_text IS NULL OR e2.entity_text IS NULL
        """

        if entity_type:
            sql += f" AND (e1.entity_type = '{entity_type}' OR e2.entity_type = '{entity_type}')"

        return self.query(sql)

    def entity_frequency(
        self,
        method: str = "llm_gpt4o_mini",
        entity_type: Optional[str] = None,
        min_mentions: int = 5,
        group_by: str = "year",
    ) -> pl.DataFrame:
        """
        Get entity mention frequency over time.

        Args:
            method: Analysis method ID.
            entity_type: Optional filter by entity type.
            min_mentions: Minimum mentions to include.
            group_by: Time grouping ("year", "month", "date").

        Returns:
            DataFrame with entity frequency by time period.
        """
        entities_path = self.results_dir / self.source / "entities" / method / "entities.parquet"

        if not entities_path.exists():
            logger.warning(f"Entity results not found: {entities_path}")
            return pl.DataFrame()

        # Determine time grouping
        if group_by == "year":
            time_expr = "YEAR(s.date)"
        elif group_by == "month":
            time_expr = "DATE_TRUNC('month', s.date)"
        else:
            time_expr = "s.date"

        sql = f"""
            SELECT 
                {time_expr} as time_period,
                e.entity_text,
                e.entity_type,
                COUNT(*) as mention_count
            FROM '{entities_path}' e
            JOIN source_lines s ON e.line_id = s.line_id
        """

        if entity_type:
            sql += f" WHERE e.entity_type = '{entity_type}'"

        sql += f"""
            GROUP BY time_period, e.entity_text, e.entity_type
            HAVING mention_count >= {min_mentions}
            ORDER BY time_period, mention_count DESC
        """

        return self.query(sql)

    def get_topic_distribution(
        self, method: str = "llm_gpt4o_mini", group_by: str = "year"
    ) -> pl.DataFrame:
        """
        Get topic distribution over time.

        Args:
            method: Analysis method ID.
            group_by: Time grouping ("year", "month", "date").

        Returns:
            DataFrame with topic counts by time period.
        """
        topics_path = self.results_dir / self.source / "topics" / method / "topics.parquet"

        if not topics_path.exists():
            logger.warning(f"Topic results not found: {topics_path}")
            return pl.DataFrame()

        if group_by == "year":
            time_expr = "YEAR(s.date)"
        elif group_by == "month":
            time_expr = "DATE_TRUNC('month', s.date)"
        else:
            time_expr = "s.date"

        sql = f"""
            SELECT 
                {time_expr} as time_period,
                t.primary_topic,
                COUNT(*) as count
            FROM '{topics_path}' t
            JOIN source_lines s ON t.line_id = s.line_id
            GROUP BY time_period, t.primary_topic
            ORDER BY time_period, count DESC
        """

        return self.query(sql)

    def close(self):
        """Close database connection."""
        self.con.close()
        logger.debug("Closed DuckDB connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_result_metadata(
    analysis_type: str,
    method_type: str,
    model_name: str,
    source: str,
    parameters: Dict[str, Any],
    line_count: int,
    duration_seconds: float,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create metadata dictionary for analysis results.

    Args:
        analysis_type: Type of analysis (e.g., "entities", "topics").
        method_type: Method type ("llm" or "traditional").
        model_name: Model identifier.
        source: Source dataset name.
        parameters: Analysis parameters/configuration.
        line_count: Number of lines processed.
        duration_seconds: Processing time in seconds.
        model_version: Optional model version.

    Returns:
        Metadata dictionary.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = f"{method_type}_{model_name}_{timestamp}".replace(".", "_").replace("-", "_")

    return {
        "analysis_id": analysis_id,
        "analysis_type": analysis_type,
        "method_type": method_type,
        "model_name": model_name,
        "model_version": model_version,
        "parameters": parameters,
        "source": source,
        "created_at": datetime.now().isoformat(),
        "line_count": line_count,
        "duration_seconds": duration_seconds,
    }
