"""
Base classes and shared state for the NiceGUI interface
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import polars as pl

from newspaper_explorer.config.base import get_config
from newspaper_explorer.utils.sources import list_available_sources, load_source_config
from newspaper_explorer.ui.nicegui.utils.data import DataLoader
from newspaper_explorer.data.utils.images import ImageIndexer


class AppState:
    """Global application state shared across all UI modules"""

    def __init__(self):
        self.config = get_config()
        self.available_sources = list_available_sources()
        self.selected_source: Optional[str] = None
        self.source_config = None
        self.data_loader: Optional[DataLoader] = None
        self.image_indexer: Optional[ImageIndexer] = None
        self.entities_df = None
        self.start_date = date(1900, 1, 1)
        self.end_date = date(2024, 12, 31)
        self.current_image_page = 1

        # Auto-load first available source
        if self.available_sources:
            self.load_source(self.available_sources[0])

    def load_source(self, source_name: str) -> None:
        """Load a source and initialize data loader and image indexer"""
        self.selected_source = source_name
        self.source_config = load_source_config(source_name)

        # Initialize data loader
        try:
            self.data_loader = DataLoader(source_name)
        except Exception:
            self.data_loader = None

        # Initialize image indexer
        try:
            self.image_indexer = ImageIndexer(source_name)
        except Exception:
            self.image_indexer = None

        # Load entities if available
        self.load_entities()

    def load_entities(self):
        """Load entity data from CSV if available"""
        if not self.selected_source:
            return None

        csv_path = Path(self.config.results_dir) / self.selected_source / "entities" / "test.csv"
        if csv_path.exists():
            self.entities_df = pl.read_csv(csv_path)
            if "Date" in self.entities_df.columns:
                # Convert Date column to datetime
                self.entities_df = self.entities_df.with_columns(
                    pl.col("Date").str.to_datetime().alias("date")
                )
        return self.entities_df

    def get_filtered_entities(self):
        """Get entities filtered by current date range"""
        if self.entities_df is None:
            return pl.DataFrame()

        # Polars filtering
        filtered = self.entities_df.filter(
            (pl.col("date") >= datetime.combine(self.start_date, datetime.min.time()))
            & (pl.col("date") <= datetime.combine(self.end_date, datetime.max.time()))
        )

        return filtered

    def get_source_stats(self) -> dict:
        """Get statistics about the current source (lightweight version for sidebar)"""
        stats = {
            "name": self.selected_source or "None",
            "newspaper_title": "Unknown",
            "years_available": "Unknown",
            "language": "Unknown",
            "total_documents": 0,
            "total_archive_size": None,
            "source_provider": None,
            "license": None,
            "info": None,
            "citation": None,
        }

        if self.source_config:
            stats["newspaper_title"] = self.source_config.metadata.newspaper_title
            stats["years_available"] = self.source_config.metadata.years_available or "Unknown"
            stats["language"] = self.source_config.metadata.language or "Unknown"
            stats["source_provider"] = self.source_config.source_provider
            stats["license"] = self.source_config.license
            stats["info"] = self.source_config.metadata.info
            stats["citation"] = self.source_config.metadata.citation

            # Calculate total archive size from parts
            if self.source_config.parts:
                total_size_gb = 0.0
                for part in self.source_config.parts:
                    if part.size:
                        # Parse size string (e.g., "1.4 GB", "375.8 MB")
                        size_str = part.size.upper()
                        if "GB" in size_str:
                            total_size_gb += float(size_str.replace("GB", "").strip())
                        elif "MB" in size_str:
                            total_size_gb += float(size_str.replace("MB", "").strip()) / 1024

                if total_size_gb > 0:
                    stats["total_archive_size"] = f"{total_size_gb:.1f} GB (compressed)"

        # Get document count from data loader
        if self.data_loader and self.data_loader.parquet_exists():
            loader_stats = self.data_loader.get_stats()
            stats["total_documents"] = loader_stats["total_files"]

        return stats

    def get_source_config(self, source_name: str):
        """Get config for any source by name"""
        return load_source_config(source_name)

    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive statistics using DataLoader"""
        if not self.data_loader or not self.data_loader.parquet_exists():
            return {
                "total_lines": 0,
                "total_files": 0,
                "total_issues": 0,
                "min_date": "N/A",
                "max_date": "N/A",
                "years": 0,
                "avg_pages": 0,
                "total_pages": 0,
            }

        return self.data_loader.get_stats()

    def get_sample_data(self, limit: int = 5) -> list:
        """Get sample data for preview"""
        if not self.data_loader or not self.data_loader.parquet_exists():
            return []

        return self.data_loader.get_sample_data(limit=limit)

    def get_analysis_results(self) -> dict:
        """Get summary of available analysis results"""
        if not self.selected_source:
            return {}

        results_path = Path(self.config.results_dir) / self.selected_source
        results = {}

        # Check each analysis type
        analysis_types = ["entities", "emotions", "topics", "keywords", "concepts", "layout"]
        for analysis_type in analysis_types:
            analysis_dir = results_path / analysis_type
            if analysis_dir.exists():
                # Count files
                parquet_files = list(analysis_dir.glob("**/*.parquet"))
                csv_files = list(analysis_dir.glob("**/*.csv"))
                total_files = len(parquet_files) + len(csv_files)

                if total_files > 0:
                    results[analysis_type] = {
                        "count": total_files,
                        "parquet": len(parquet_files),
                        "csv": len(csv_files),
                    }

        return results

    def get_sample_images(self, limit: int = 6) -> list:
        """Get sample images from the source, one from each different year with variety in page numbers"""
        if not self.selected_source:
            return []

        # Try to use image index first (faster and has better variety)
        if self.image_indexer:
            index = self.image_indexer.load_index()
            if index is not None and len(index) > 0:
                samples = self.image_indexer.get_sample_images(limit=limit, spread_years=True)
                if len(samples) > 0:
                    return samples["image_path"].to_list()

        # Fallback to filesystem scan
        images_path = Path(self.config.data_dir) / "raw" / self.selected_source / "images"
        if not images_path.exists():
            return []

        # Find all year directories
        year_dirs = sorted([d for d in images_path.iterdir() if d.is_dir() and d.name.isdigit()])

        if not year_dirs:
            return []

        # Get one image from each year, spreading across available years
        sample_images = []
        step = max(1, len(year_dirs) // limit)

        for i, year_dir in enumerate(year_dirs[::step][:limit]):
            # Find images in this year
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                images = sorted(list(year_dir.rglob(ext)))
                if images:
                    # Pick a different image based on index to get variety in page numbers
                    # Use modulo to cycle through available images
                    img_index = i % len(images)
                    sample_images.append(str(images[img_index].relative_to(images_path)))
                    break

            if len(sample_images) >= limit:
                break

        return sample_images

    def get_image_path(self, relative_path: str) -> Path:
        """Get full path to an image"""
        if not self.selected_source:
            return Path()
        return Path(self.config.data_dir) / "raw" / self.selected_source / "images" / relative_path
