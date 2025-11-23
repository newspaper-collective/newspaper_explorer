"""
Query engine using DuckDB for efficient analysis of newspaper data.

Provides SQL interface to source data and analysis results without
loading entire files into memory.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import polars as pl

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    DuckDB-based query engine for newspaper data.

    Enables SQL queries across Parquet files (source data + analysis results)
    without loading entire files into memory.

    Example:
        ```python
        from newspaper_explorer.analyze.query.engine import QueryEngine

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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from the source parquet file.

        Returns:
            Dictionary with statistics including line counts, date ranges,
            and image counts.
        """
        if not self.source_parquet.exists():
            return {
                "total_lines": 0,
                "total_files": 0,
                "total_issues": 0,
                "total_blocks": 0,
                "min_date": "N/A",
                "max_date": "N/A",
                "years": 0,
                "avg_pages": 0,
                "total_pages": 0,
                "total_images": 0,
            }

        try:
            result = self.con.execute(
                f"""
                SELECT 
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT filename) as total_files,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT text_block_id) as total_blocks,
                    MIN(date) as min_date,
                    MAX(date) as max_date,
                    AVG(page_count) as avg_pages,
                    COUNT(DISTINCT page_id) as total_pages
                FROM read_parquet('{self.source_parquet}')
            """
            ).fetchone()

            min_date = result[4].strftime("%Y-%m-%d") if result[4] else "N/A"
            max_date = result[5].strftime("%Y-%m-%d") if result[5] else "N/A"

            # Calculate years
            years = 0
            if result[4] and result[5]:
                years = result[5].year - result[4].year + 1

            # Get image count from image_index.parquet if it exists
            total_images = 0
            image_index_path = self.source_parquet.parent.parent / "image_index.parquet"
            if image_index_path.exists():
                try:
                    img_result = self.con.execute(
                        f"SELECT COUNT(*) FROM read_parquet('{image_index_path}')"
                    ).fetchone()
                    total_images = img_result[0] if img_result else 0
                except Exception:
                    pass

            return {
                "total_lines": result[0] or 0,
                "total_files": result[1] or 0,
                "total_issues": result[2] or 0,
                "total_blocks": result[3] or 0,
                "min_date": min_date,
                "max_date": max_date,
                "years": years,
                "avg_pages": result[6] or 0,
                "total_pages": result[7] or 0,
                "total_images": total_images,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_lines": 0,
                "total_files": 0,
                "total_issues": 0,
                "total_blocks": 0,
                "total_images": 0,
                "min_date": "N/A",
                "max_date": "N/A",
                "years": 0,
                "avg_pages": 0,
                "total_pages": 0,
            }

    def get_sample_data(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get random sample data for preview.

        Args:
            limit: Number of samples to return.

        Returns:
            List of sample records.
        """
        if not self.source_parquet.exists():
            return []

        try:
            result = self.con.execute(
                f"""
                SELECT 
                    date,
                    SUBSTR(text, 1, 100) || '...' as text,
                    page_number,
                    newspaper_title,
                    issue_number
                FROM read_parquet('{self.source_parquet}')
                WHERE text IS NOT NULL AND LENGTH(text) > 20
                ORDER BY RANDOM()
                LIMIT {limit}
            """
            ).fetchall()

            return [
                {
                    "date": row[0].strftime("%Y-%m-%d") if row[0] else "N/A",
                    "text": row[1],
                    "page": row[2] or "N/A",
                    "title": row[3] or "N/A",
                    "issue": row[4] or "N/A",
                }
                for row in result
            ]
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return []

    def get_date_range_stats(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get statistics for a specific date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Statistics dictionary.
        """
        if not self.source_parquet.exists():
            return {"total_lines": 0, "total_issues": 0, "total_pages": 0}

        try:
            result = self.con.execute(
                f"""
                SELECT 
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT page_id) as total_pages
                FROM read_parquet('{self.source_parquet}')
                WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
            ).fetchone()

            return {
                "total_lines": result[0] or 0,
                "total_issues": result[1] or 0,
                "total_pages": result[2] or 0,
            }
        except Exception as e:
            logger.error(f"Error getting date range stats: {e}")
            return {"total_lines": 0, "total_issues": 0, "total_pages": 0}

    def search_text_simple(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simple text search with pagination (for UI).

        Args:
            query: Search query.
            limit: Maximum results to return.
            offset: Offset for pagination.
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).

        Returns:
            List of matching records with image paths.
        """
        if not self.source_parquet.exists():
            return []

        try:
            # Build WHERE clause
            where_clause = f"LOWER(text) LIKE LOWER('%{query}%')"
            if start_date:
                where_clause += f" AND date >= '{start_date}'"
            if end_date:
                where_clause += f" AND date <= '{end_date}'"

            # Simple LIKE search
            result = self.con.execute(
                f"""
                SELECT 
                    date,
                    text,
                    page_number,
                    newspaper_title,
                    issue_number,
                    line_id,
                    page_id,
                    text_block_id,
                    x,
                    y,
                    width,
                    height
                FROM read_parquet('{self.source_parquet}')
                WHERE {where_clause}
                ORDER BY date DESC
                LIMIT {limit}
                OFFSET {offset}
            """
            ).fetchall()

            # Get image paths
            image_index_path = self.source_parquet.parent.parent / "image_index.parquet"
            image_map = {}

            if image_index_path.exists() and result:
                page_ids = [f"'{row[6]}'" for row in result if row[6]]
                if page_ids:
                    page_ids_str = ",".join(page_ids)
                    try:
                        img_result = self.con.execute(
                            f"""
                            SELECT page_id, image_path 
                            FROM read_parquet('{image_index_path}')
                            WHERE page_id IN ({page_ids_str})
                            """
                        ).fetchall()
                        image_map = {row[0]: row[1] for row in img_result}
                    except Exception as e:
                        logger.warning(f"Error fetching image paths: {e}")

            return [
                {
                    "date": row[0].strftime("%Y-%m-%d") if row[0] else "N/A",
                    "text": row[1],
                    "page": row[2] or "N/A",
                    "title": row[3] or "N/A",
                    "issue": row[4] or "N/A",
                    "line_id": row[5],
                    "page_id": row[6],
                    "text_block_id": row[7],
                    "x": row[8],
                    "y": row[9],
                    "width": row[10],
                    "height": row[11],
                    "image_path": image_map.get(row[6]),
                }
                for row in result
            ]
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []

    def search_text_count(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        Get total count of search results.

        Args:
            query: Search query.
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).

        Returns:
            Total count of matching records.
        """
        if not self.source_parquet.exists():
            return 0

        try:
            # Build WHERE clause
            where_clause = f"LOWER(text) LIKE LOWER('%{query}%')"
            if start_date:
                where_clause += f" AND date >= '{start_date}'"
            if end_date:
                where_clause += f" AND date <= '{end_date}'"

            result = self.con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{self.source_parquet}')
                WHERE {where_clause}
            """
            ).fetchone()

            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error counting search results: {e}")
            return 0

    def get_yearly_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics aggregated by year.

        Returns:
            List of yearly statistics.
        """
        if not self.source_parquet.exists():
            return []

        try:
            result = self.con.execute(
                f"""
                SELECT 
                    year,
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT filename) as total_files
                FROM read_parquet('{self.source_parquet}')
                GROUP BY year
                ORDER BY year
            """
            ).fetchall()

            return [
                {
                    "year": row[0],
                    "total_lines": row[1],
                    "total_issues": row[2],
                    "total_files": row[3],
                }
                for row in result
            ]
        except Exception as e:
            logger.error(f"Error getting yearly stats: {e}")
            return []

    def get_monthly_distribution(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get distribution of documents by month.

        Args:
            year: Optional year filter.

        Returns:
            List of monthly counts.
        """
        if not self.source_parquet.exists():
            return []

        year_filter = f"WHERE year = {year}" if year else ""

        try:
            result = self.con.execute(
                f"""
                SELECT 
                    month,
                    COUNT(DISTINCT issue_id) as issue_count
                FROM read_parquet('{self.source_parquet}')
                {year_filter}
                GROUP BY month
                ORDER BY month
            """
            ).fetchall()

            return [{"month": row[0], "issue_count": row[1]} for row in result]
        except Exception as e:
            logger.error(f"Error getting monthly distribution: {e}")
            return []

    def execute_custom_query(self, query_sql: str) -> List[tuple]:
        """
        Execute a custom SQL query on the parquet file.

        Args:
            query_sql: SQL query string.

        Returns:
            Query results as list of tuples.
        """
        if not self.source_parquet.exists():
            return []

        try:
            # Replace placeholder table name with actual parquet path
            query_sql = query_sql.replace("{{parquet}}", f"read_parquet('{self.source_parquet}')")
            result = self.con.execute(query_sql).fetchall()
            return result
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            return []

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
