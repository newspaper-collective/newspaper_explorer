"""
Query engine using DuckDB for efficient analysis of newspaper data.

Provides SQL interface to source data and analysis results without
loading entire files into memory.
"""

import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

import duckdb
import polars as pl

from newspaper_explorer.config.base import get_config

# DuckDB exception types for proper error handling
DuckDBError = duckdb.Error

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
        *,
        in_memory: bool = True,
    ) -> None:
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
        self.source_parquet = config.parsed_dir / source / "lines.parquet"

        # Create views for common queries
        self._create_views()

    def _create_views(self) -> None:
        """Create common views for easier querying."""
        if not self.source_parquet.exists():
            logger.warning(f"Source parquet not found: {self.source_parquet}")
            return

        # Create view for source data
        parquet_path = str(self.source_parquet).replace("'", "''")
        self.con.execute(
            f"CREATE OR REPLACE VIEW source_lines AS SELECT * FROM read_parquet('{parquet_path}')"
        )
        logger.debug("Created source_lines view")

    def query(self, sql: str, params: Optional[list[Any]] = None) -> pl.DataFrame:
        """
        Execute SQL query and return Polars DataFrame.

        Args:
            sql: SQL query string.
            params: Optional list of parameters for parameterized query.

        Returns:
            Query result as Polars DataFrame.
        """
        result = self.con.execute(sql, params).df() if params else self.con.execute(sql).df()

        return pl.from_pandas(result)

    def get_line(self, line_id: str) -> Optional[dict[str, Any]]:
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

        sql = """
            SELECT
                e.entity_text,
                e.entity_type,
                e.confidence,
                s.line_id,
                s.text,
                s.date,
                s.filename,
                s.text_block_id
            FROM read_parquet(?) e
            JOIN source_lines s ON e.line_id = s.line_id
            WHERE e.entity_text = ?
        """
        params = [str(entities_path), entity_name]

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

        sql = """
            SELECT
                COALESCE(e1.line_id, e2.line_id) as line_id,
                e1.entity_text as method1_entity,
                e1.entity_type as method1_type,
                e2.entity_text as method2_entity,
                e2.entity_type as method2_type,
                s.text,
                s.date
            FROM read_parquet(?) e1
            FULL OUTER JOIN read_parquet(?) e2
                ON e1.line_id = e2.line_id
                AND e1.entity_text = e2.entity_text
            JOIN source_lines s
                ON COALESCE(e1.line_id, e2.line_id) = s.line_id
            WHERE e1.entity_text IS NULL OR e2.entity_text IS NULL
        """
        params = [str(path1), str(path2)]

        if entity_type:
            sql += " AND (e1.entity_type = ? OR e2.entity_type = ?)"
            params.extend([entity_type, entity_type])

        return self.query(sql, params=params)

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

        # Whitelist for time grouping to prevent SQL injection
        time_grouping_map = {
            "year": "YEAR(s.date)",
            "month": "DATE_TRUNC('month', s.date)",
            "date": "s.date",
        }

        if group_by not in time_grouping_map:
            logger.warning(f"Invalid group_by value: {group_by}. Using 'year'.")
            group_by = "year"

        time_expr = time_grouping_map[group_by]

        # Build SQL with validated time expression (safe from injection)
        sql = (
            """
            SELECT
                """
            + time_expr
            + """ as time_period,
                e.entity_text,
                e.entity_type,
                COUNT(*) as mention_count
            FROM read_parquet(?) e
            JOIN source_lines s ON e.line_id = s.line_id
        """
        )
        params: list[Any] = [str(entities_path)]

        if entity_type:
            sql += " WHERE e.entity_type = ?"
            params.append(entity_type)

        sql += """
            GROUP BY time_period, e.entity_text, e.entity_type
            HAVING mention_count >= ?
            ORDER BY time_period, mention_count DESC
        """
        params.append(min_mentions)

        return self.query(sql, params)

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

        # Whitelist for time grouping to prevent SQL injection
        time_grouping_map = {
            "year": "YEAR(s.date)",
            "month": "DATE_TRUNC('month', s.date)",
            "date": "s.date",
        }

        if group_by not in time_grouping_map:
            logger.warning(f"Invalid group_by value: {group_by}. Using 'year'.")
            group_by = "year"

        time_expr = time_grouping_map[group_by]

        # Build SQL with validated time expression (safe from injection)
        sql = (
            """
            SELECT
                """
            + time_expr
            + """ as time_period,
                t.primary_topic,
                COUNT(*) as count
            FROM read_parquet(?) t
            JOIN source_lines s ON t.line_id = s.line_id
            GROUP BY time_period, t.primary_topic
            ORDER BY time_period, count DESC
        """
        )

        return self.query(sql, [str(topics_path)])

    def get_stats(self) -> dict[str, Any]:
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
                """
                SELECT
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT filename) as total_files,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT text_block_id) as total_blocks,
                    MIN(date) as min_date,
                    MAX(date) as max_date,
                    AVG(page_count) as avg_pages,
                    COUNT(DISTINCT page_id) as total_pages
                FROM read_parquet(?)
            """,
                [str(self.source_parquet)],
            ).fetchone()

            # Handle case where query returns None
            if result is None:
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

            # Now we can safely index into result
            min_date_obj = result[4]
            max_date_obj = result[5]

            min_date = min_date_obj.strftime("%Y-%m-%d") if min_date_obj else "N/A"
            max_date = max_date_obj.strftime("%Y-%m-%d") if max_date_obj else "N/A"

            # Calculate years
            years = 0
            if min_date_obj and max_date_obj:
                years = max_date_obj.year - min_date_obj.year + 1

            # Get image count from image_index.parquet if it exists
            total_images = 0
            image_index_path = self.source_parquet.parent.parent / "image_index.parquet"
            if image_index_path.exists():
                try:
                    img_result = self.con.execute(
                        "SELECT COUNT(*) FROM read_parquet(?)", [str(image_index_path)]
                    ).fetchone()
                    if img_result is not None:
                        total_images = img_result[0] or 0
                except DuckDBError:
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
        except DuckDBError as e:
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

    def get_sample_data(self, limit: int = 5) -> list[dict[str, Any]]:
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
                """
                SELECT
                    date,
                    SUBSTR(text, 1, 100) || '...' as text,
                    page_number,
                    newspaper_title,
                    issue_number
                FROM read_parquet(?)
                WHERE text IS NOT NULL AND LENGTH(text) > 20
                ORDER BY RANDOM()
                LIMIT ?
            """,
                [str(self.source_parquet), limit],
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
        except DuckDBError as e:
            logger.error(f"Error getting sample data: {e}")
            return []

    def get_date_range_stats(self, start_date: str, end_date: str) -> dict[str, Any]:
        """
        Get statistics for a specific date range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Statistics dictionary with total_lines, total_issues, total_pages.
        """
        if not self.source_parquet.exists():
            logger.warning(f"Source parquet not found: {self.source_parquet}")
            return self._empty_date_stats()

        try:
            result = self.con.execute(
                """
                SELECT
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT page_id) as total_pages
                FROM read_parquet(?)
                WHERE date >= ? AND date <= ?
                """,
                [str(self.source_parquet), start_date, end_date],
            ).fetchone()

            if not result:
                return self._empty_date_stats()

            return {
                "total_lines": result[0] or 0,
                "total_issues": result[1] or 0,
                "total_pages": result[2] or 0,
            }

        except DuckDBError as e:
            logger.error(f"DuckDB error getting date range stats: {e}")
            return self._empty_date_stats()
        except (IndexError, TypeError) as e:
            logger.error(f"Error processing date range stats result: {e}")
            return self._empty_date_stats()

    def _empty_date_stats(self) -> dict[str, Any]:
        """Return empty date range statistics."""
        return {
            "total_lines": 0,
            "total_issues": 0,
            "total_pages": 0,
        }

    def search_text(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Text search with pagination and image paths.

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
            # Build parameterized query dynamically based on filters
            base_query = """
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
                FROM read_parquet(?)
                WHERE LOWER(text) LIKE LOWER(?)
            """
            params: list[Any] = [str(self.source_parquet), f"%{query}%"]

            if start_date:
                base_query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                base_query += " AND date <= ?"
                params.append(end_date)

            base_query += """
                ORDER BY date DESC
                LIMIT ?
                OFFSET ?
            """
            params.extend([limit, offset])

            # Execute main search query
            result = self.con.execute(base_query, params).fetchall()

            # Get image paths
            image_index_path = self.source_parquet.parent.parent / "image_index.parquet"
            image_map: dict[Any, Any] = {}

            if image_index_path.exists() and result:
                page_ids = [row[6] for row in result if row[6]]
                if page_ids:
                    # Validate page_ids are safe types (defense in depth)
                    if not all(isinstance(pid, (str, int)) for pid in page_ids):
                        logger.warning("Invalid page_id types detected, skipping image lookup")
                    else:
                        # Use DuckDB's list parameter support for safe IN clause
                        img_result = self.con.execute(
                            """
                            SELECT page_id, image_path
                            FROM read_parquet(?)
                            WHERE page_id = ANY(?)
                        """,
                            [str(image_index_path), page_ids],
                        ).fetchall()
                        image_map = {row[0]: row[1] for row in img_result}

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
        except DuckDBError as e:
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
            # Build SQL with validated conditions (safe from injection)
            sql = """
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE LOWER(text) LIKE LOWER(?)
            """
            params: list[Any] = [str(self.source_parquet), f"%{query}%"]

            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND date <= ?"
                params.append(end_date)

            result = self.con.execute(sql, params).fetchone()

            return result[0] if result else 0
        except DuckDBError as e:
            logger.error(f"Error counting search results: {e}")
            return 0

    def get_yearly_stats(self) -> list[dict[str, Any]]:
        """
        Get statistics aggregated by year.

        Returns:
            List of yearly statistics.
        """
        if not self.source_parquet.exists():
            return []

        try:
            result = self.con.execute(
                """
                SELECT
                    year,
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT filename) as total_files
                FROM read_parquet(?)
                GROUP BY year
                ORDER BY year
            """,
                [str(self.source_parquet)],
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
        except DuckDBError as e:
            logger.error(f"Error getting yearly stats: {e}")
            return []

    def get_monthly_distribution(self, year: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Get distribution of documents by month.

        Args:
            year: Optional year filter.

        Returns:
            List of monthly counts.
        """
        if not self.source_parquet.exists():
            return []

        try:
            if year:
                result = self.con.execute(
                    """
                    SELECT
                        month,
                        COUNT(DISTINCT issue_id) as issue_count
                    FROM read_parquet(?)
                    WHERE year = ?
                    GROUP BY month
                    ORDER BY month
                """,
                    [str(self.source_parquet), year],
                ).fetchall()
            else:
                result = self.con.execute(
                    """
                    SELECT
                        month,
                        COUNT(DISTINCT issue_id) as issue_count
                    FROM read_parquet(?)
                    GROUP BY month
                    ORDER BY month
                """,
                    [str(self.source_parquet)],
                ).fetchall()

            return [{"month": row[0], "issue_count": row[1]} for row in result]
        except DuckDBError as e:
            logger.error(f"Error getting monthly distribution: {e}")
            return []

    def execute_custom_query(self, query_sql: str) -> list[Any]:
        """
        Execute a custom SQL query on the parquet file.

        Args:
            query_sql: SQL query string with '{{parquet}}' placeholder.
                       Example: "SELECT * FROM {{parquet}} WHERE year > 1900"

        Returns:
            Query results as list of row tuples.

        Raises:
            ValueError: If query doesn't contain {{parquet}} placeholder.
        """
        if not self.source_parquet.exists():
            return []

        # Validate placeholder exists to prevent accidental raw SQL
        if "{{parquet}}" not in query_sql:
            raise ValueError("Query must contain {{parquet}} placeholder")

        try:
            # Replace placeholder with DuckDB's parameterized function
            query_sql = query_sql.replace("{{parquet}}", "read_parquet(?)")
            return self.con.execute(query_sql, [str(self.source_parquet)]).fetchall()

        except DuckDBError as e:
            logger.error(f"Error executing custom query: {e}")
            return []

    def close(self) -> None:
        """Close database connection."""
        self.con.close()
        logger.debug("Closed DuckDB connection")

    def __enter__(self) -> "QueryEngine":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit."""
        self.close()
