"""
Data loading and DuckDB utilities for the NiceGUI interface
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import duckdb

from newspaper_explorer.config.base import get_config


class DataLoader:
    """Helper class for loading and querying newspaper data with DuckDB"""

    def __init__(self, source_name: str):
        """
        Initialize data loader for a specific source

        Args:
            source_name: Name of the newspaper source
        """
        self.source_name = source_name
        self.config = get_config()
        self.db = duckdb.connect()

        # Set up parquet path
        self.parquet_path = (
            Path(self.config.data_dir)
            / "raw"
            / source_name
            / "text"
            / f"{source_name}_lines.parquet"
        )

    def parquet_exists(self) -> bool:
        """Check if parquet file exists"""
        return self.parquet_path.exists()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from the parquet file

        Returns:
            Dictionary with statistics
        """
        if not self.parquet_exists():
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
            }

        try:
            result = self.db.execute(
                f"""
                SELECT 
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT filename) as total_files,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT text_block_id) as total_blocks,
                    MIN(date) as min_date,
                    MAX(date) as max_date,
                    AVG(page_count) as avg_pages,
                    SUM(DISTINCT page_count) as total_pages
                FROM read_parquet('{self.parquet_path}')
            """
            ).fetchone()

            min_date = result[4].strftime("%Y-%m-%d") if result[4] else "N/A"
            max_date = result[5].strftime("%Y-%m-%d") if result[5] else "N/A"

            # Calculate years
            years = 0
            if result[4] and result[5]:
                years = result[5].year - result[4].year + 1

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
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
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
            }

    def get_sample_data(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get random sample data for preview

        Args:
            limit: Number of samples to return

        Returns:
            List of sample records
        """
        if not self.parquet_exists():
            return []

        try:
            result = self.db.execute(
                f"""
                SELECT 
                    date,
                    SUBSTR(text, 1, 100) || '...' as text,
                    page_number,
                    newspaper_title,
                    issue_number
                FROM read_parquet('{self.parquet_path}')
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
            print(f"Error getting sample data: {e}")
            return []

    def get_date_range_stats(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get statistics for a specific date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Statistics dictionary
        """
        if not self.parquet_exists():
            return {"total_lines": 0, "total_issues": 0, "total_pages": 0}

        try:
            result = self.db.execute(
                f"""
                SELECT 
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT page_id) as total_pages
                FROM read_parquet('{self.parquet_path}')
                WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
            ).fetchone()

            return {
                "total_lines": result[0] or 0,
                "total_issues": result[1] or 0,
                "total_pages": result[2] or 0,
            }
        except Exception as e:
            print(f"Error getting date range stats: {e}")
            return {"total_lines": 0, "total_issues": 0, "total_pages": 0}

    def search_text(self, query: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search for text in the parquet file

        Args:
            query: Search query
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            List of matching records
        """
        if not self.parquet_exists():
            return []

        try:
            # Simple LIKE search (can be enhanced with full-text search)
            result = self.db.execute(
                f"""
                SELECT 
                    date,
                    text,
                    page_number,
                    newspaper_title,
                    issue_number,
                    line_id
                FROM read_parquet('{self.parquet_path}')
                WHERE LOWER(text) LIKE LOWER('%{query}%')
                ORDER BY date DESC
                LIMIT {limit}
                OFFSET {offset}
            """
            ).fetchall()

            return [
                {
                    "date": row[0].strftime("%Y-%m-%d") if row[0] else "N/A",
                    "text": row[1],
                    "page": row[2] or "N/A",
                    "title": row[3] or "N/A",
                    "issue": row[4] or "N/A",
                    "line_id": row[5],
                }
                for row in result
            ]
        except Exception as e:
            print(f"Error searching text: {e}")
            return []

    def get_yearly_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics aggregated by year

        Returns:
            List of yearly statistics
        """
        if not self.parquet_exists():
            return []

        try:
            result = self.db.execute(
                f"""
                SELECT 
                    year,
                    COUNT(*) as total_lines,
                    COUNT(DISTINCT issue_id) as total_issues,
                    COUNT(DISTINCT filename) as total_files
                FROM read_parquet('{self.parquet_path}')
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
            print(f"Error getting yearly stats: {e}")
            return []

    def get_monthly_distribution(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get distribution of documents by month

        Args:
            year: Optional year filter

        Returns:
            List of monthly counts
        """
        if not self.parquet_exists():
            return []

        year_filter = f"WHERE year = {year}" if year else ""

        try:
            result = self.db.execute(
                f"""
                SELECT 
                    month,
                    COUNT(DISTINCT issue_id) as issue_count
                FROM read_parquet('{self.parquet_path}')
                {year_filter}
                GROUP BY month
                ORDER BY month
            """
            ).fetchall()

            return [{"month": row[0], "issue_count": row[1]} for row in result]
        except Exception as e:
            print(f"Error getting monthly distribution: {e}")
            return []

    def execute_custom_query(self, query: str) -> List[tuple]:
        """
        Execute a custom SQL query on the parquet file

        Args:
            query: SQL query string (should reference the parquet file as a table)

        Returns:
            Query results as list of tuples
        """
        if not self.parquet_exists():
            return []

        try:
            # Replace placeholder table name with actual parquet path
            query = query.replace("{{parquet}}", f"read_parquet('{self.parquet_path}')")
            result = self.db.execute(query).fetchall()
            return result
        except Exception as e:
            print(f"Error executing custom query: {e}")
            return []
