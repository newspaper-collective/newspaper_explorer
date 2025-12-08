"""
Source management endpoints
"""

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from newspaper_explorer.analyze.query.engine import QueryEngine
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.sources import list_available_sources, load_source_config
from newspaper_explorer.models.api.sources import AnalysisResultSummary, SourceInfo, SourceStats

router = APIRouter()


@router.get("/", response_model=List[str])
async def list_sources():
    """List all available sources"""
    return list_available_sources()


@router.get("/{source_name}", response_model=SourceInfo)
async def get_source_info(source_name: str):
    """Get detailed information about a source"""
    try:
        config = load_source_config(source_name)
        app_config = get_config()

        # Check what data is available
        raw_path = Path(app_config.data_dir) / "raw" / source_name
        results_path = Path(app_config.results_dir) / source_name

        has_text = (raw_path / "text").exists()
        has_images = (raw_path / "images").exists()

        # Calculate total archive size from parts
        total_archive_size = None
        if config.parts:
            total_size_gb = 0.0
            for part in config.parts:
                if part.size:
                    # Parse size string (e.g., "1.4 GB", "375.8 MB")
                    size_str = part.size.upper()
                    if "GB" in size_str:
                        total_size_gb += float(size_str.replace("GB", "").strip())
                    elif "MB" in size_str:
                        total_size_gb += float(size_str.replace("MB", "").strip()) / 1024

            if total_size_gb > 0:
                total_archive_size = f"{total_size_gb:.1f} GB (compressed)"

        # Scan analysis results and check for actual files
        analysis_results = {}
        analysis_types = ["entities", "emotions", "topics", "keywords", "concepts", "layout"]
        for analysis_type in analysis_types:
            analysis_dir = results_path / analysis_type
            if analysis_dir.exists():
                # Count files
                parquet_files = list(analysis_dir.glob("**/*.parquet"))
                csv_files = list(analysis_dir.glob("**/*.csv"))
                total_files = len(parquet_files) + len(csv_files)

                if total_files > 0:
                    # Try to find and read metadata from JSON files
                    metadata = None
                    # Look for common metadata file names
                    for metadata_name in [
                        f"{analysis_type}.json",
                        "metadata.json",
                        "entities.json",
                        "layout.json",
                    ]:
                        metadata_files = list(analysis_dir.glob(f"**/{metadata_name}"))
                        if metadata_files:
                            try:
                                with open(metadata_files[0], "r") as f:
                                    metadata = json.load(f)
                                break
                            except:
                                pass

                    analysis_results[analysis_type] = AnalysisResultSummary(
                        count=total_files,
                        parquet=len(parquet_files),
                        csv=len(csv_files),
                        metadata=metadata,
                    )

        # Set has_* flags based on actual file counts, not just directory existence
        has_entities = "entities" in analysis_results
        has_keywords = "keywords" in analysis_results
        has_layout = "layout" in analysis_results
        has_topics = "topics" in analysis_results
        has_emotions = "emotions" in analysis_results
        has_concepts = "concepts" in analysis_results

        # Calculate XML file stats
        xml_file_count = None
        xml_total_size = None
        if has_text:
            try:
                text_path = raw_path / "text"
                xml_files = list(text_path.rglob("*.xml"))
                xml_file_count = len(xml_files)
                if xml_file_count > 0:
                    total_bytes = sum(f.stat().st_size for f in xml_files)
                    total_gb = total_bytes / (1024**3)
                    if total_gb >= 1.0:
                        xml_total_size = f"{total_gb:.1f} GB"
                    else:
                        total_mb = total_bytes / (1024**2)
                        xml_total_size = f"{total_mb:.0f} MB"
            except Exception:
                pass

        # Calculate parquet size
        parquet_size = None
        parquet_path = raw_path / "text" / f"{source_name}_lines.parquet"
        if parquet_path.exists():
            try:
                size_bytes = parquet_path.stat().st_size
                size_gb = size_bytes / (1024**3)
                if size_gb >= 1.0:
                    parquet_size = f"{size_gb:.1f} GB"
                else:
                    size_mb = size_bytes / (1024**2)
                    parquet_size = f"{size_mb:.0f} MB"
            except Exception:
                pass

        # Get image statistics if images exist
        image_size: Optional[str] = None
        image_count: Optional[int] = None
        if has_images:
            try:
                from newspaper_explorer.data.indexing.image_index import ImageIndexer

                indexer = ImageIndexer(source_name)
                index = indexer.load_index()
                if index is not None and len(index) > 0:
                    stats = indexer.get_stats()
                    image_size = f"{stats['total_size_gb']:.1f} GB"
                    total_images = stats.get("total_images")
                    if total_images is not None:
                        image_count = int(total_images)
            except Exception:
                # If image indexer fails, ignore
                pass

        # Build metadata dict with both nested metadata and top-level fields
        metadata_dict = config.metadata.model_dump() if config.metadata else {}
        # Add top-level fields that should be in metadata for frontend
        metadata_dict["source_provider"] = config.source_provider
        metadata_dict["license"] = config.license
        metadata_dict["description"] = config.description

        return SourceInfo(
            name=source_name,
            dataset_name=config.dataset_name,
            data_type=config.data_type,
            metadata=metadata_dict,
            loading=config.loading.model_dump() if config.loading else {},
            has_text=has_text,
            has_entities=has_entities,
            has_keywords=has_keywords,
            has_layout=has_layout,
            has_topics=has_topics,
            has_emotions=has_emotions,
            has_concepts=has_concepts,
            has_images=has_images,
            total_archive_size=total_archive_size,
            xml_file_count=xml_file_count,
            xml_total_size=xml_total_size,
            parquet_size=parquet_size,
            image_size=image_size,
            image_count=image_count,
            analysis_results=analysis_results,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found")


@router.get("/{source_name}/stats", response_model=SourceStats)
async def get_source_stats(source_name: str):
    """Get statistics for a source"""
    try:
        qe = QueryEngine(source=source_name, in_memory=True)

        if not qe.source_parquet.exists():
            raise HTTPException(status_code=404, detail="No data available for this source")

        # Get stats from DuckDB
        stats = qe.get_stats()

        # Extract date range
        date_range = (stats["min_date"], stats["max_date"])

        # Calculate years available
        years_available = []
        if stats["min_date"] != "N/A" and stats["max_date"] != "N/A":
            from datetime import datetime

            min_year = datetime.strptime(stats["min_date"], "%Y-%m-%d").year
            max_year = datetime.strptime(stats["max_date"], "%Y-%m-%d").year
            years_available = list(range(min_year, max_year + 1))

        return SourceStats(
            total_issues=stats["total_issues"],
            total_pages=stats["total_pages"],
            total_lines=stats["total_lines"],
            total_blocks=stats["total_blocks"],
            total_images=stats.get("total_images", 0),
            date_range=date_range,
            years_available=years_available,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
