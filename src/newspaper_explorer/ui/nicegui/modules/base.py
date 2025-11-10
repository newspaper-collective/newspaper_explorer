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
        self.keywords_df = None
        self.layout_df = None  # Layout detection results
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

        # Load keywords if available
        self.load_keywords()

        # Load layout if available
        self.load_layout()

    def get_available_entity_files(self) -> list:
        """
        Get list of available entity result files for the current source.
        Scans subdirectories for entities.parquet with metadata.json.

        Returns:
            List of tuples (display_name, file_path) for each available entity file
        """
        if not self.selected_source:
            return []

        entities_dir = Path(self.config.results_dir) / self.selected_source / "entities"
        if not entities_dir.exists():
            return []

        result_files = []

        # Scan subdirectories for entities.parquet and metadata.json
        for subdir in entities_dir.iterdir():
            if not subdir.is_dir():
                continue

            parquet_file = subdir / "entities.parquet"
            metadata_file = subdir / "metadata.json"

            if not parquet_file.exists():
                continue

            # Try to create display name from metadata
            import json

            display_name = subdir.name
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Create nice display name from metadata
                    method = metadata.get("method_type", "")
                    model = metadata.get("model_name", "")
                    created = metadata.get("created_at", "")

                    if created:
                        # Extract date part
                        date_part = created.split("T")[0] if "T" in created else created[:10]
                    else:
                        date_part = ""

                    if method and model:
                        display_name = f"{method} - {model}"
                        if date_part:
                            display_name += f" ({date_part})"
                    elif method:
                        display_name = method
                        if date_part:
                            display_name += f" ({date_part})"
                except:
                    pass

            result_files.append((display_name, str(parquet_file)))

        # Sort by display name
        result_files.sort(key=lambda x: x[0])
        return result_files

    def load_entities(self, file_path: Optional[str] = None):
        """
        Load entity data from parquet file in new format.

        Expected columns: entity_text, entity_type, line_id, page_id, etc.
        Will extract date from page_id automatically.

        Args:
            file_path: Path to entity parquet file
        """
        if not self.selected_source:
            return None

        # Use provided path or look for first available result
        if file_path is None:
            available = self.get_available_entity_files()
            if not available:
                self.entities_df = None
                return None
            file_path = available[0][1]  # Use first available file

        path = Path(file_path)
        if not path.exists():
            self.entities_df = None
            return None

        # Load parquet file
        try:
            self.entities_df = pl.read_parquet(str(path))

            # Extract date from page_id
            # page_id format: {source}_{YYYY-MM-DD}_{page}_{subpage}
            self.entities_df = self.entities_df.with_columns(
                [
                    pl.col("page_id")
                    .str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
                    .str.to_date("%Y-%m-%d")
                    .cast(pl.Datetime)
                    .alias("date")
                ]
            )
            # Don't fail, but visualizations may not work properly

        except Exception as e:
            print(f"Error loading entity file {path}: {e}")
            import traceback

            traceback.print_exc()
            self.entities_df = None

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

    def get_available_keyword_files(self) -> list:
        """
        Get list of available keyword result files for the current source.
        Scans subdirectories for keywords.parquet with metadata.json.

        Returns:
            List of tuples (display_name, file_path) for each available keyword file
        """
        if not self.selected_source:
            return []

        keywords_dir = Path(self.config.results_dir) / self.selected_source / "keywords"
        if not keywords_dir.exists():
            return []

        result_files = []

        # New format: subdirectories with keywords.parquet and metadata.json
        for subdir in keywords_dir.iterdir():
            if subdir.is_dir():
                parquet_file = subdir / "keywords.parquet"
                metadata_file = subdir / "metadata.json"

                if parquet_file.exists():
                    # Try to create display name from metadata
                    display_name = subdir.name
                    if metadata_file.exists():
                        try:
                            import json

                            with open(metadata_file, "r", encoding="utf-8") as f:
                                metadata = json.load(f)

                            # Create nice display name from metadata
                            method = metadata.get("method_type", "")
                            model = metadata.get("model_name", "")
                            created = metadata.get("created_at", "")

                            if created:
                                date_part = (
                                    created.split("T")[0] if "T" in created else created[:10]
                                )
                            else:
                                date_part = ""

                            if method and model:
                                display_name = f"{method} - {model}"
                                if date_part:
                                    display_name += f" ({date_part})"
                            elif method:
                                display_name = method
                                if date_part:
                                    display_name += f" ({date_part})"
                        except:
                            pass

                    result_files.append((display_name, str(parquet_file)))

        # Sort by display name
        result_files.sort(key=lambda x: x[0])
        return result_files

    def load_keywords(self, file_path: Optional[str] = None):
        """
        Load keyword data from parquet file

        Args:
            file_path: Optional path to keyword file. If None, looks for default files
        """
        if not self.selected_source:
            return None

        # Use provided path or try to find a default
        if file_path is None:
            keywords_dir = Path(self.config.results_dir) / self.selected_source / "keywords"
            if not keywords_dir.exists():
                self.keywords_df = None
                return None

            # Try to find any parquet file (prefer test files)
            possible_files = [
                "test_keybert_by_date.parquet",
                "test_keybert_optimized.parquet",
                "test_keybert.parquet",
            ]
            for filename in possible_files:
                path = keywords_dir / filename
                if path.exists():
                    file_path = str(path)
                    break

            # If no default found, try first parquet file
            if file_path is None:
                parquet_files = list(keywords_dir.glob("*.parquet"))
                if parquet_files:
                    file_path = str(parquet_files[0])
                else:
                    self.keywords_df = None
                    return None

        path = Path(file_path)
        if not path.exists():
            self.keywords_df = None
            return None

        # Load parquet file
        try:
            self.keywords_df = pl.read_parquet(str(path))
        except Exception as e:
            print(f"Error loading keyword file {path}: {e}")
            self.keywords_df = None

        return self.keywords_df

    def get_filtered_keywords(self):
        """
        Get keywords filtered by current date range (if doc_id contains date info)

        Note: Filtering by date requires doc_id to contain date information.
        For now, returns all keywords as date filtering may not be applicable.
        """
        if self.keywords_df is None:
            return pl.DataFrame()

        # For now, return all keywords
        # TODO: Implement date filtering if doc_id contains parseable date info
        return self.keywords_df

    def get_available_layout_files(self) -> list:
        """
        Get list of available layout detection result files for the current source.
        Scans subdirectories for layout.parquet with layout.json.

        Returns:
            List of tuples (display_name, file_path, metadata_dict) for each available layout file
        """
        if not self.selected_source:
            return []

        layout_dir = Path(self.config.results_dir) / self.selected_source / "layout"
        if not layout_dir.exists():
            return []

        result_files = []

        # Scan subdirectories for layout.parquet and layout.json
        for subdir in layout_dir.iterdir():
            if not subdir.is_dir():
                continue

            parquet_file = subdir / "layout.parquet"
            metadata_file = subdir / "layout.json"

            if not parquet_file.exists():
                continue

            # Try to create display name from metadata
            import json

            display_name = subdir.name
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Create nice display name from metadata
                    model = metadata.get("model_name", "")
                    created = metadata.get("created_at", "")
                    num_detections = metadata.get("output_data", {}).get("num_detections", 0)

                    if created:
                        # Extract date part
                        date_part = created.split("T")[0] if "T" in created else created[:10]
                    else:
                        date_part = ""

                    if model and date_part:
                        display_name = f"{model} ({date_part}) - {num_detections} detections"
                    elif model:
                        display_name = f"{model} - {num_detections} detections"
                except:
                    pass

            result_files.append((display_name, str(parquet_file), metadata))

        # Sort by display name (most recent first due to timestamp in name)
        result_files.sort(key=lambda x: x[0], reverse=True)
        return result_files

    def load_layout(self, file_path: Optional[str] = None):
        """
        Load layout detection data from parquet file.

        Args:
            file_path: Path to layout parquet file
        """
        if not self.selected_source:
            return None

        # Use provided path or look for first available result
        if file_path is None:
            available = self.get_available_layout_files()
            if not available:
                self.layout_df = None
                return None
            file_path = available[0][1]  # Use first (most recent) available file

        path = Path(file_path)
        if not path.exists():
            self.layout_df = None
            return None

        # Load parquet file
        try:
            self.layout_df = pl.read_parquet(str(path))

            # Extract date from page_id (format: {source}_{YYYY-MM-DD}_{issue}_{daily}_{page})
            self.layout_df = self.layout_df.with_columns(
                [
                    pl.col("page_id")
                    .str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
                    .str.to_date("%Y-%m-%d")
                    .cast(pl.Datetime)
                    .alias("date")
                ]
            )

        except Exception as e:
            print(f"Error loading layout file {path}: {e}")
            import traceback

            traceback.print_exc()
            self.layout_df = None

        return self.layout_df

    def get_filtered_layout(self):
        """Get layout detections filtered by current date range"""
        if self.layout_df is None:
            return pl.DataFrame()

        # Polars filtering
        filtered = self.layout_df.filter(
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
