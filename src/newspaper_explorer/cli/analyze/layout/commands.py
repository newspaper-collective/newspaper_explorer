"""
CLI commands for layout analysis.

Provides commands to:
- Detect layout elements (headlines, pictures, tables)
- Visualize detections for debugging
- Extract picture regions with captions
- Match headlines to OCR text
- Build articles from headlines and text blocks

Usage:
    newspaper-explorer analyze layout detect --source der_tag --year 1902
    newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001
    newspaper-explorer analyze layout extract-pictures --source der_tag --year 1902
    newspaper-explorer analyze layout match-headlines --source der_tag --year 1902
    newspaper-explorer analyze layout build-articles --source der_tag --year 1902
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time

import click
from tqdm import tqdm

from newspaper_explorer.analyze.layout.article_builder import ArticleBuilder
from newspaper_explorer.analyze.layout.detection import LayoutDetector
from newspaper_explorer.analyze.layout.headline_matcher import HeadlineMatcher
from newspaper_explorer.analyze.layout.visualizer import LayoutVisualizer
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.ingest.loader import DataIngester
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.models.data.metadata import AnalysisMetadata

logger = logging.getLogger(__name__)


def get_text_data_path(source: str, text_data: str | None = None) -> Path:
    """
    Get path to text data (raw or preprocessed).

    Args:
        source: Source name
        text_data: Optional path or preset name. Options:
            - None: Use raw data (default)
            - "raw": Explicit raw data
            - "preprocessed": Use default preprocessed data
            - Path: Custom parquet file path

    Returns:
        Path to parquet file
    """
    config = get_config()

    if text_data is None or text_data == "raw":
        # Default: raw line-level data
        return config.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"
    elif text_data == "preprocessed":
        # Default preprocessed data
        return config.data_dir / "processed" / source / "text" / "textblocks_processed.parquet"
    else:
        # Custom path
        return Path(text_data)


@click.group(name="layout")
def layout_group():
    """Layout analysis commands for newspaper images."""
    pass


@layout_group.command()
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--model-size",
    type=click.Choice(["nano", "small", "medium"]),
    default="medium",
    help="YOLOv11 model size (default: medium)",
)
@click.option(
    "--device",
    default="cuda:0",
    help="Device for inference: 'cuda:0', 'cuda:1', etc., or 'cpu' (default: cuda:0)",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for inference (default: 32)",
)
@click.option(
    "--conf-threshold",
    type=float,
    default=0.2,
    help="Confidence threshold (default: 0.2)",
)
@click.option(
    "--year",
    type=int,
    help="Process only specific year",
)
@click.option(
    "--limit",
    type=int,
    help="Limit number of pages to process",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Skip already processed pages (default: yes)",
)
def detect(source, model_size, device, batch_size, conf_threshold, year, limit, resume):
    """
    Detect layout elements in newspaper images.

    Detects 11 element types: Caption, Footnote, Formula, List-item,
    Page-footer, Page-header, Picture, Section-header, Table, Text, Title.

    Example:
        newspaper-explorer analyze layout detect --source der_tag --year 1902
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("Layout Detection with YOLOv11")
    click.echo(f"{'=' * 60}\n")

    config = get_config()

    # Find image files
    click.echo(f"Finding images for source: {source}")
    images_dir = config.data_dir / "raw" / source / "images"

    if not images_dir.exists():
        click.echo(f"✗ Images directory not found: {images_dir}", err=True)
        click.echo("  Run 'newspaper-explorer data download-images' first", err=True)
        return

    # Collect image paths
    image_paths = []
    search_pattern = f"{year}/**/*.jpg" if year else "**/*.jpg"

    for img_path in images_dir.glob(search_pattern):
        image_paths.append(img_path)
        if limit and len(image_paths) >= limit:
            break

    if not image_paths:
        click.echo(f"✗ No images found in {images_dir}", err=True)
        return

    click.echo(f"✓ Found {len(image_paths)} images")

    # Generate analysis_id for this run
    model_short = f"yolo11{model_size[0]}"  # nano -> yolo11n, medium -> yolo11m
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = f"{model_short}_doclaynet_{timestamp}"

    # Check for existing results and implement resume
    output_dir = config.results_dir / source / "layout" / analysis_id
    output_file = output_dir / "layout.parquet"

    processed_pages = set()
    if resume and output_file.exists():
        import polars as pl

        existing_df = pl.read_parquet(output_file)
        processed_pages = set(existing_df["page_id"].unique().to_list())
        click.echo(f"✓ Resume mode: {len(processed_pages)} pages already processed")

        # Filter out already processed images
        from newspaper_explorer.analyze.layout.detection import LayoutDetector

        temp_detector = LayoutDetector(
            model_size=model_size, device="cpu"
        )  # Lightweight for ID generation
        original_count = len(image_paths)
        image_paths = [
            img
            for img in image_paths
            if temp_detector._generate_page_id(img) not in processed_pages
        ]
        skipped = original_count - len(image_paths)
        if skipped > 0:
            click.echo(f"✓ Skipping {skipped} already processed pages")

        if not image_paths:
            click.echo("✓ All pages already processed!", err=False)
            return

    # Check GPU availability
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                click.echo(f"✓ Found {gpu_count} GPU(s) available")

                if gpu_count > 1:
                    click.echo(f"  Note: YOLO only uses single GPU (device {device})")
                    click.echo(f"  Multi-GPU parallelism not supported by YOLO's predict() method")
                    click.echo(f"  Recommendation: Use largest possible batch size on single GPU")
            else:
                click.echo("⚠ CUDA requested but no GPUs available, falling back to CPU", err=True)
                device = "cpu"
        except ImportError:
            click.echo("⚠ PyTorch not found, cannot check GPU availability", err=True)

    # Initialize detector
    click.echo(f"\nInitializing YOLOv11 {model_size} model...")

    from newspaper_explorer.analyze.layout.detection import LayoutDetector

    detector = LayoutDetector(
        model_size=model_size,
        device=device,
        batch_size=batch_size,
        conf_threshold=conf_threshold,
        source_name=source,  # Pass source name for proper ID generation
    )

    # Run detection
    click.echo("\nDetecting layout elements...")
    # Don't pass page_ids - let detector generate unique IDs from paths

    # Start timing
    start_time = time.time()

    # Temporarily suppress detector's INFO logs to avoid interfering with tqdm
    detector_logger = logging.getLogger("newspaper_explorer.analyze.layout.detection")
    original_level = detector_logger.level
    detector_logger.setLevel(logging.WARNING)

    try:
        with tqdm(total=len(image_paths), desc="Processing pages") as pbar:
            # Update progress in batches
            results = []
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]

                batch_results = detector.detect_batch(batch_paths, page_ids=None)
                results.extend(batch_results)

                pbar.update(len(batch_paths))
    finally:
        # Restore original log level
        detector_logger.setLevel(original_level)

    # Calculate duration
    duration = time.time() - start_time
    completed_at = datetime.now().isoformat()

    # Flatten all detections into rows
    import polars as pl

    all_detections = []
    skipped_count = 0

    for page_layout in results:
        for det in page_layout.detections:
            # Skip detections with problematic data
            # (None values or malformed source_ids that can't be reliably typed)
            try:
                # Ensure source_id is consistently typed (string or None, not mixed)
                source_id = str(det.source_id) if det.source_id is not None else None
                issue_id = str(det.issue_id) if det.issue_id is not None else None

                all_detections.append(
                    {
                        "detection_id": det.detection_id,
                        "page_id": det.page_id,
                        "source_id": source_id,
                        "issue_id": issue_id,
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "bbox_x1": det.bbox.x1,
                        "bbox_y1": det.bbox.y1,
                        "bbox_x2": det.bbox.x2,
                        "bbox_y2": det.bbox.y2,
                        "bbox_width": det.bbox.width,
                        "bbox_height": det.bbox.height,
                        "image_path": page_layout.image_path,
                    }
                )
            except Exception as e:
                logger.debug(f"Skipping detection with problematic data: {e}")
                skipped_count += 1

    if skipped_count > 0:
        click.echo(f"⚠ Skipped {skipped_count} detections with incomplete data", err=True)

    # Create DataFrame and save with metadata
    if all_detections:
        df = pl.DataFrame(all_detections)

        # If resuming, merge with existing data
        if resume and output_file.exists():
            existing_df = pl.read_parquet(output_file)
            df = pl.concat([existing_df, df])
            click.echo(f"\nAppended {len(all_detections)} new detections")

        # Create metadata
        metadata = AnalysisMetadata(
            analysis_id=analysis_id,
            analysis_type="layout",
            method_type="yolov11",
            model_name=f"yolo11{model_size[0]}_doc_layout",
            model_version="DocLayNet",
            source=source,
            completed_at=completed_at,
            duration_seconds=duration,
            granularity="page",  # Layout detection operates at page level
            parameters={
                "model_size": model_size,
                "device": device,
                "batch_size": batch_size,
                "conf_threshold": conf_threshold,
                "year_filter": year,
                "classes": [
                    "Caption",
                    "Footnote",
                    "Formula",
                    "List-item",
                    "Page-footer",
                    "Page-header",
                    "Picture",
                    "Section-header",
                    "Table",
                    "Text",
                    "Title",
                ],
            },
            input_data={
                "num_images": len(image_paths),
                "year_filter": year,
            },
            output_data={
                "num_pages": df["page_id"].n_unique(),
                "num_detections": len(df),
                "detections_by_class": df.group_by("class_name")
                .agg(pl.len().alias("count"))
                .to_dicts(),
            },
        )

        # Save using standard pattern
        results_dict = save_analysis_results(
            results_df=df,
            metadata=metadata,
            results_base_dir=config.results_dir / source,
        )

        click.echo(f"\nSaved {len(df)} detections to: {results_dict['results_path']}")
        click.echo(f"Saved metadata to: {results_dict['metadata_path']}")
        click.echo(f"Analysis ID: {analysis_id}")
    else:
        click.echo("\nNo detections to save", err=True)

    # Statistics
    total_detections = sum(len(r.detections) for r in results)

    # Count by class type
    if all_detections:
        df = pl.DataFrame(all_detections)
        class_counts = (
            df.group_by("class_name").agg(pl.len().alias("count")).sort("count", descending=True)
        )

        click.echo(f"\n{'=' * 60}")
        click.echo("Detection Complete!")
        click.echo(f"{'=' * 60}")
        click.echo(f"Pages processed: {len(results)}")
        click.echo(f"Total detections: {total_detections}")
        click.echo("\nDetections by class:")
        for row in class_counts.iter_rows(named=True):
            click.echo(f"  {row['class_name']}: {row['count']}")
        click.echo(f"{'=' * 60}\n")
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo("Detection Complete!")
        click.echo(f"{'=' * 60}")
        click.echo(f"Pages processed: {len(results)}")
        click.echo(f"Total detections: 0")
        click.echo(f"{'=' * 60}\n")


@layout_group.command()
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--year",
    type=int,
    help="Process only specific year",
)
@click.option(
    "--save-crops/--no-save-crops",
    default=True,
    help="Save cropped image files (default: yes)",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Skip already processed pages (default: yes)",
)
@click.option(
    "--exclude-top-percent",
    type=float,
    default=None,
    help="Exclude regions in top X%% of page (e.g., 10 for headers)",
)
@click.option(
    "--exclude-bottom-percent",
    type=float,
    default=None,
    help="Exclude regions in bottom X%% of page (e.g., 5 for footers)",
)
@click.option(
    "--min-height",
    type=int,
    default=None,
    help="Minimum region height in pixels",
)
@click.option(
    "--min-width",
    type=int,
    default=None,
    help="Minimum region width in pixels",
)
def extract_pictures(
    source,
    year,
    save_crops,
    resume,
    exclude_top_percent,
    exclude_bottom_percent,
    min_height,
    min_width,
):
    """
    Extract picture regions from newspaper pages (without caption matching).

    Extracts detected picture regions and saves crops. For caption matching,
    use the 'match-captions' command after extraction.

    Coordinate-based filtering options help exclude common false positives like
    newspaper headers (--exclude-top-percent) or page numbers (--exclude-bottom-percent).

    Examples:
        # Basic extraction
        newspaper-explorer analyze layout extract-pictures --source der_tag --year 1902

        # Exclude top 15% (headers) and small regions
        newspaper-explorer analyze layout extract-pictures --source der_tag --year 1902 \\
            --exclude-top-percent 15 --min-height 100 --min-width 100
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("Image Extraction")
    click.echo(f"{'=' * 60}\n")

    config = get_config()

    # Load detection results
    detections_dir = config.results_dir / source / "layout"
    if not detections_dir.exists():
        click.echo(f"✗ No layout detections found in {detections_dir}", err=True)
        click.echo("  Run 'newspaper-explorer analyze layout detect' first", err=True)
        return

    click.echo(f"Loading detection results from {detections_dir}")

    # Find detection files
    import json

    from newspaper_explorer.models.analysis.layout import PageLayout

    detection_files = list(detections_dir.glob("*_layout.json"))
    if year:
        detection_files = [
            f for f in detection_files if f"/{year}/" in str(f) or f"_{year}_" in f.stem
        ]

    if not detection_files:
        click.echo(f"✗ No detection files found", err=True)
        return

    click.echo(f"✓ Found {len(detection_files)} detection files")

    # Initialize extractor with filtering options
    from newspaper_explorer.analyze.layout.region_extraction import RegionExtractor

    extractor = RegionExtractor(
        padding=5,
        exclude_top_percent=exclude_top_percent,
        exclude_bottom_percent=exclude_bottom_percent,
        min_region_height=min_height,
        min_region_width=min_width,
    )

    # Show filter configuration
    if any([exclude_top_percent, exclude_bottom_percent, min_height, min_width]):
        click.echo("\nFiltering configuration:")
        if exclude_top_percent:
            click.echo(f"  ✓ Excluding top {exclude_top_percent}% of page (headers)")
        if exclude_bottom_percent:
            click.echo(f"  ✓ Excluding bottom {exclude_bottom_percent}% of page (footers)")
        if min_height:
            click.echo(f"  ✓ Minimum height: {min_height}px")
        if min_width:
            click.echo(f"  ✓ Minimum width: {min_width}px")

    # Extract images
    output_dir = config.results_dir / source / "layout" / "images"
    metadata_path = output_dir.parent / f"{source}_images_metadata.parquet"

    # Check for existing metadata and determine processed pages
    import polars as pl

    processed_pages = set()
    if resume and metadata_path.exists():
        existing_df = pl.read_parquet(metadata_path)
        processed_pages = set(existing_df["page_id"].unique().to_list())
        click.echo(f"✓ Resume mode: {len(processed_pages)} pages already processed")

    total_images = 0
    skipped_pages = 0

    click.echo("\nExtracting images...")
    for idx, det_file in enumerate(tqdm(detection_files, desc="Processing pages")):
        with open(det_file, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        page_id = page_data["page_id"]

        # Skip if already processed (resume mode)
        if resume and page_id in processed_pages:
            skipped_pages += 1
            continue

        # Reconstruct PageLayout (simplified)
        from newspaper_explorer.models.analysis.layout import BoundingBox, Detection

        page_layout = PageLayout(
            page_id=page_id,
            image_path=page_data["image_path"],
            detections=[],
        )

        # Collect images
        images = []
        for img_data in page_data.get("images", []):
            det = Detection(
                detection_id=img_data["detection_id"],
                class_name=img_data["class_name"],
                confidence=img_data["confidence"],
                bbox=BoundingBox(**img_data["bbox"]),
                page_id=page_id,
            )
            images.append(det)

        # Extract and crop pictures
        if images:
            extracted = extractor.extract_regions(
                detections=images,
                page_layout=page_layout,
                output_dir=output_dir,
                region_type="picture",
            )

            # Save metadata incrementally (always append in resume mode)
            mode = (
                "append"
                if (resume and metadata_path.exists())
                else "overwrite"
                if idx == 0
                else "append"
            )
            extractor.save_region_metadata(extracted, metadata_path, mode=mode)
            total_images += len(extracted)

    click.echo(f"\n{'=' * 60}")
    click.echo("Image Extraction Complete!")
    click.echo(f"{'=' * 60}")
    click.echo(f"Total images extracted: {total_images}")
    if resume and skipped_pages > 0:
        click.echo(f"Skipped pages (already processed): {skipped_pages}")
    if save_crops:
        click.echo(f"Crops saved to: {output_dir}")
    click.echo(f"Metadata saved to: {metadata_path}")
    click.echo(f"\n💡 To match captions, run:")
    click.echo(f"   newspaper-explorer analyze layout match-captions --source {source}")
    click.echo(f"{'=' * 60}\n")


@layout_group.command()
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--year",
    type=int,
    help="Process only specific year",
)
@click.option(
    "--caption-position",
    type=click.Choice(["below", "above", "both"]),
    default="below",
    help="Where to look for captions (default: below)",
)
@click.option(
    "--search-radius",
    type=int,
    default=150,
    help="Max distance to search for captions in pixels (default: 150)",
)
@click.option(
    "--overlap-threshold",
    type=float,
    default=0.3,
    help="IoU threshold for caption-text matching (default: 0.3)",
)
@click.option(
    "--text-data",
    help="Text data source: 'raw' (default), 'preprocessed', or path to parquet file",
)
def match_captions(source, year, caption_position, search_radius, overlap_threshold, text_data):
    """
    Match captions to extracted images.

    Uses spatial proximity and OCR text extraction to match caption
    detections to nearby images.

    Example:
        newspaper-explorer analyze layout match-captions --source der_tag --year 1902
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("Caption Matching")
    click.echo(f"{'=' * 60}\n")

    config = get_config()

    # Load detection results
    detections_dir = config.results_dir / source / "layout"
    if not detections_dir.exists():
        click.echo(f"✗ No layout detections found in {detections_dir}", err=True)
        click.echo("  Run 'newspaper-explorer analyze layout detect' first", err=True)
        return

    click.echo(f"Loading detection results from {detections_dir}")

    # Find detection files
    import json

    from newspaper_explorer.models.analysis.layout import BoundingBox, Detection, PageLayout

    detection_files = list(detections_dir.glob("*_layout.json"))
    if year:
        detection_files = [
            f for f in detection_files if f"/{year}/" in str(f) or f"_{year}_" in f.stem
        ]

    if not detection_files:
        click.echo(f"✗ No detection files found", err=True)
        return

    click.echo(f"✓ Found {len(detection_files)} detection files")

    # Load parsed text for caption OCR extraction
    lines_path = get_text_data_path(source, text_data)
    if not lines_path.exists():
        click.echo(f"✗ Text data not found: {lines_path}", err=True)
        click.echo("  Run 'newspaper-explorer data parse' or 'preprocess' first", err=True)
        return

    import polars as pl

    click.echo(f"Loading text data from: {lines_path}")
    lines_df = pl.read_parquet(lines_path)
    if year:
        lines_df = lines_df.filter(pl.col("year") == year)

    click.echo(f"✓ Loaded {len(lines_df)} text lines for caption extraction")

    # Initialize proximity matcher (handles text extraction + spatial matching)
    from newspaper_explorer.analyze.layout.region_matching import ProximityMatcher

    # Map old caption_position values to new relative_position
    position_map = {"both": "any", "below": "below", "above": "above"}
    relative_pos = position_map.get(caption_position, "below")

    matcher = ProximityMatcher(
        search_radius=search_radius,
        relative_position=relative_pos,  # type: ignore
        overlap_threshold=overlap_threshold,
    )

    # Match captions
    all_matched = []

    click.echo("\nMatching captions...")
    for det_file in tqdm(detection_files, desc="Processing pages"):
        with open(det_file, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        page_id = page_data["page_id"]

        # Collect images and captions
        images = []
        for img_data in page_data.get("images", []):
            det = Detection(
                detection_id=img_data["detection_id"],
                class_name=img_data["class_name"],
                confidence=img_data["confidence"],
                bbox=BoundingBox(**img_data["bbox"]),
                page_id=page_id,
            )
            images.append(det)

        captions = []
        for cap_data in page_data.get("captions", []):
            det = Detection(
                detection_id=cap_data["detection_id"],
                class_name=cap_data["class_name"],
                confidence=cap_data["confidence"],
                bbox=BoundingBox(**cap_data["bbox"]),
                page_id=page_id,
            )
            captions.append(det)

        # Match captions to images (handles text extraction + spatial matching)
        if images:
            matches = matcher.match_elements(
                source_elements=images,
                target_elements=captions,
                lines_df=lines_df,
                page_id=page_id,
                extract_text=True,
            )
            images_with_captions = matcher.apply_matches(matches, target_attr="caption")
            all_matched.extend(images_with_captions)

    # Save results
    output_dir = config.results_dir / source / "layout"
    output_path = output_dir / f"{source}_image_captions.parquet"

    if all_matched:
        captions_data = [
            {
                "image_id": img.detection_id,
                "page_id": img.page_id,
                "caption_text": img.caption_text,
                "caption_id": img.caption.detection_id if img.caption else None,
                "caption_confidence": img.caption.confidence if img.caption else None,
            }
            for img in all_matched
        ]
        import polars as pl

        df = pl.DataFrame(captions_data)
        df.write_parquet(output_path)

    click.echo(f"\n{'=' * 60}")
    click.echo("Caption Matching Complete!")
    click.echo(f"{'=' * 60}")
    click.echo(f"Total images: {len(all_matched)}")
    click.echo(f"Images with captions: {sum(1 for img in all_matched if img.caption_text)}")
    click.echo(f"Results saved to: {output_path}")
    click.echo(f"{'=' * 60}\n")


@layout_group.command("extract-caption-text")
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--year",
    type=int,
    help="Process only specific year",
)
@click.option(
    "--run-id",
    help="Specific layout detection run ID to use",
)
@click.option(
    "--min-iou",
    type=float,
    default=0.1,
    help="Minimum IoU threshold for text line intersection (default: 0.1)",
)
@click.option(
    "--text-data",
    help="Text data source: 'raw' (default), 'preprocessed', or path to parquet file",
)
def extract_caption_text(source, year, run_id, min_iou, text_data):
    """
    Extract text content from detected caption regions using TextLinker.

    This command:
    1. Loads layout detections (Caption class)
    2. Loads ALTO text lines from parquet
    3. Uses image index for proper coordinate scaling
    4. Matches caption bounding boxes to text lines via TextLinker
    5. Saves enriched caption data with extracted text

    Uses the image index to get original image dimensions for accurate
    coordinate scaling between layout detection space and ALTO space.

    Example:
        newspaper-explorer analyze layout extract-caption-text --source der_tag --year 1902
        newspaper-explorer analyze layout extract-caption-text --source der_tag --run-id 20241121_143022
    """
    import cv2
    import polars as pl

    from newspaper_explorer.analyze.layout.text_linker import TextLinker
    from newspaper_explorer.models.analysis.layout import BoundingBox, Detection

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("Caption Text Extraction with Coordinate Normalization")
    click.echo(f"{'=' * 60}\n")

    config = get_config()

    # Load detection results
    layout_dir = config.results_dir / source / "layout"
    if not layout_dir.exists():
        click.echo(f"✗ Layout results not found: {layout_dir}", err=True)
        return

    # Find the layout run to use
    if run_id:
        layout_path = layout_dir / run_id / "layout.parquet"
        output_dir = layout_dir / run_id
    else:
        # Use most recent
        result_dirs = [
            d for d in layout_dir.iterdir() if d.is_dir() and (d / "layout.parquet").exists()
        ]
        if not result_dirs:
            click.echo(f"✗ No layout results found in {layout_dir}", err=True)
            return
        result_dirs.sort(reverse=True)
        layout_path = result_dirs[0] / "layout.parquet"
        output_dir = result_dirs[0]
        run_id = result_dirs[0].name

    if not layout_path.exists():
        click.echo(f"✗ Layout data not found: {layout_path}", err=True)
        return

    click.echo(f"Using layout results: {run_id}")
    click.echo(f"Loading from: {layout_path}")

    # Load layout detections
    detections_df = pl.read_parquet(layout_path)

    # Filter for captions only
    captions_df = detections_df.filter(pl.col("class_name") == "Caption")

    if year:
        # Extract year from page_id and filter
        captions_df = captions_df.with_columns(
            pl.col("page_id").str.extract(r"_(\d{4})-", 1).cast(pl.Int32).alias("year")
        ).filter(pl.col("year") == year)

    if len(captions_df) == 0:
        click.echo(f"✗ No caption detections found", err=True)
        return

    click.echo(f"✓ Found {len(captions_df)} caption detections")

    # Load text data
    lines_path = get_text_data_path(source, text_data)
    if not lines_path.exists():
        click.echo(f"✗ Text data not found: {lines_path}", err=True)
        click.echo("  Run 'newspaper-explorer data parse' first", err=True)
        return

    click.echo(f"Loading text lines from: {lines_path}")
    lines_df = pl.read_parquet(lines_path)

    if year:
        lines_df = lines_df.filter(pl.col("year") == year)

    click.echo(f"✓ Loaded {len(lines_df)} text lines")

    # Initialize TextLinker with image index for coordinate scaling
    click.echo(f"\nInitializing TextLinker with image index...")

    # Suppress TextLinker logging during progress bar
    linker_logger = logging.getLogger("newspaper_explorer.analyze.layout.text_linker")
    original_level = linker_logger.level
    linker_logger.setLevel(logging.WARNING)

    linker = TextLinker(overlap_threshold=min_iou, source_name=source)

    if linker.image_index is None:
        click.echo(
            f"⚠ Warning: Image index not found. Run: newspaper-explorer data index-images --source {source}",
            err=True,
        )
        click.echo(f"  Coordinate scaling may be inaccurate!", err=True)

    # Process captions
    results = []
    captions_with_text = 0

    click.echo(f"\nProcessing {len(captions_df)} captions...")

    for cap_row in tqdm(
        captions_df.iter_rows(named=True), total=len(captions_df), desc="Extracting text"
    ):
        page_id = cap_row["page_id"]
        detection_id = cap_row["detection_id"]
        image_path = cap_row["image_path"]

        # Get layout image dimensions
        if not image_path:
            logger.warning(f"No image path for caption {detection_id}")
            continue

        layout_img = cv2.imread(image_path)
        if layout_img is None:
            logger.warning(f"Could not load image: {image_path}")
            continue

        layout_height, layout_width = layout_img.shape[:2]

        # Create Detection object
        detection = Detection(
            detection_id=detection_id,
            class_name="Caption",
            confidence=cap_row["confidence"],
            bbox=BoundingBox(
                x1=cap_row["bbox_x1"],
                y1=cap_row["bbox_y1"],
                x2=cap_row["bbox_x2"],
                y2=cap_row["bbox_y2"],
            ),
            page_id=page_id,
            image_path=image_path,
        )

        # Link to text with coordinate scaling
        linked = linker.link_detections_to_text(
            detections=[detection],
            lines_df=lines_df,
            page_id=page_id,
            layout_width=layout_width,
            layout_height=layout_height,
        )

        # Extract results
        caption_text = linked[0].text_content if linked else None
        num_lines = len(linked[0].alto_elements) if linked and linked[0].alto_elements else 0

        if caption_text:
            captions_with_text += 1

        results.append(
            {
                "detection_id": detection_id,
                "page_id": page_id,
                "caption_text": caption_text,
                "num_lines": num_lines,
            }
        )

    # Restore original logging level
    linker_logger.setLevel(original_level)

    # Create results DataFrame
    if results:
        results_df = pl.DataFrame(results)

        # Save to output directory
        output_path = output_dir / "caption_text_enriched.parquet"
        results_df.write_parquet(output_path)

        click.echo(f"\n{'=' * 60}")
        click.echo("Caption Text Extraction Complete!")
        click.echo(f"{'=' * 60}")
        click.echo(f"Total captions: {len(results)}")
        click.echo(
            f"Captions with text: {captions_with_text} ({captions_with_text / len(results) * 100:.1f}%)"
        )
        click.echo(f"Results saved to: {output_path}")
        click.echo(f"{'=' * 60}\n")

        # Show sample
        if captions_with_text > 0:
            click.echo("\nSample extracted captions:")
            sample = results_df.filter(pl.col("caption_text").is_not_null()).head(5)
            for row in sample.iter_rows(named=True):
                text_preview = (
                    row["caption_text"][:80] + "..."
                    if len(row["caption_text"]) > 80
                    else row["caption_text"]
                )
                click.echo(f"  [{row['page_id']}] {text_preview}")
                click.echo(f"    Lines: {row['num_lines']}")
    else:
        click.echo(f"\n✗ No caption text extracted", err=True)


@layout_group.command()
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--year",
    type=int,
    help="Process only specific year",
)
@click.option(
    "--overlap-threshold",
    type=float,
    default=0.3,
    help="Minimum IoU for matching (default: 0.3)",
)
@click.option(
    "--text-data",
    help="Text data source: 'raw' (default), 'preprocessed', or path to parquet file",
)
def match_headlines(source, year, overlap_threshold, text_data):
    """
    Match detected headlines to OCR text from ALTO XML.

    Links headline bounding boxes to actual text content by finding
    overlapping text blocks in ALTO XML files.

    Example:
        newspaper-explorer analyze layout match-headlines --source der_tag --year 1902
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("Headline Matching to OCR Text")
    click.echo(f"{'=' * 60}\n")

    config = get_config()

    # Load parsed lines
    lines_path = get_text_data_path(source, text_data)
    click.echo(f"Loading text data from: {lines_path}")

    if not lines_path.exists():
        click.echo(f"✗ Parsed text not found: {lines_path}", err=True)
        click.echo("  Run 'newspaper-explorer data parse' first", err=True)
        return

    import polars as pl

    lines_df = pl.read_parquet(lines_path)

    if year:
        lines_df = lines_df.filter(pl.col("year") == year)

    click.echo(f"✓ Loaded {len(lines_df)} text lines")

    # Load detection results
    detections_dir = config.results_dir / source / "layout"
    detection_files = list(detections_dir.glob("*_layout.json"))

    if year:
        detection_files = [
            f for f in detection_files if f"/{year}/" in str(f) or f"_{year}_" in f.stem
        ]

    click.echo(f"✓ Found {len(detection_files)} detection files")

    # Initialize matcher
    matcher = HeadlineMatcher(overlap_threshold=overlap_threshold)

    # Match headlines
    all_headlines = []

    click.echo("\nMatching headlines...")
    for det_file in tqdm(detection_files, desc="Processing pages"):
        # Load detection
        import json

        with open(det_file, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        # Reconstruct PageLayout (simplified - headlines only)
        from newspaper_explorer.models.analysis.layout import BoundingBox, Detection, PageLayout

        page_layout = PageLayout(
            page_id=page_data["page_id"],
            image_path=page_data["image_path"],
            detections=[],
            year=page_data.get("year", 0),
        )

        headlines_list = []
        for hl_data in page_data.get("headlines", []):
            det = Detection(
                detection_id=hl_data["detection_id"],
                class_name=hl_data["class_name"],
                confidence=hl_data["confidence"],
                bbox=BoundingBox(**hl_data["bbox"]),
                page_id=page_data["page_id"],
            )
            headlines_list.append(det)

        page_layout.detections.extend(headlines_list)

        # Match headlines (using DataFrame, not ALTO XML directly)
        headlines = matcher.match_headlines(page_layout, lines_df)
        all_headlines.extend(headlines)

    # Save matched headlines
    output_dir = config.results_dir / source / "layout"
    output_path = output_dir / f"{source}_headlines.parquet"

    if all_headlines:
        headlines_data = [hl.model_dump(mode="json", exclude_none=True) for hl in all_headlines]
        import polars as pl

        df = pl.DataFrame(headlines_data)
        df.write_parquet(output_path)

        # Save metadata.json
        metadata_file = output_dir / f"{source}_headlines_metadata.json"
        metadata = {
            "analysis_type": "layout",
            "method_type": "headline_matching",
            "model_name": "coordinate_overlap",
            "model_version": None,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "overlap_threshold": overlap_threshold,
                "text_data": text_data or "raw",
                "year_filter": year,
            },
            "total_pages": len(detection_files),
            "total_headlines": len(all_headlines),
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    click.echo(f"\n{'=' * 60}")
    click.echo("Headline Matching Complete!")
    click.echo(f"{'=' * 60}")
    click.echo(f"Total headlines matched: {len(all_headlines)}")
    click.echo(f"Results saved to: {output_path}")
    click.echo(f"{'=' * 60}\n")


@layout_group.command()
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--page-id",
    help="Specific page ID to visualize (e.g., '1902_01_01_001')",
)
@click.option(
    "--year",
    type=int,
    help="Process specific year (creates visualizations for all pages)",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Limit number of pages when visualizing by year (default: 10)",
)
@click.option(
    "--element-types",
    multiple=True,
    type=click.Choice(["title", "picture", "caption", "table", "text", "all"]),
    default=["all"],
    help="Element types to show (can specify multiple)",
)
@click.option(
    "--comparison/--single",
    default=False,
    help="Create comparison view with separate panels for each element type",
)
@click.option(
    "--show-linked-text",
    is_flag=True,
    default=False,
    help="Show matched OCR text below detections (requires prior text matching via match-captions/match-headlines)",
)
def visualize(source, page_id, year, limit, element_types, comparison, show_linked_text):
    """
    Visualize detected layout elements for debugging.

    Creates annotated images showing detected regions with bounding boxes,
    labels, and confidence scores.

    Examples:
        # Visualize specific page
        newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001

        # Visualize first 10 pages of a year
        newspaper-explorer analyze layout visualize --source der_tag --year 1902 --limit 10

        # Show only headlines and images
        newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 \\
            --element-types title --element-types picture

        # Show with linked OCR text (after running match-captions/match-headlines)
        newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 --show-linked-text

        # Create comparison view
        newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 --comparison
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("Layout Visualization")
    click.echo(f"{'=' * 60}\n")

    config = get_config()

    # Load detection results from Parquet
    detections_dir = config.results_dir / source / "layout"
    detections_file = detections_dir / f"{source}_layout_detections.parquet"

    if not detections_file.exists():
        click.echo(f"✗ No layout detections found: {detections_file}", err=True)
        click.echo("  Run 'newspaper-explorer analyze layout detect' first", err=True)
        return

    click.echo(f"Loading detections from {detections_file}")
    import polars as pl

    from newspaper_explorer.models.analysis.layout import BoundingBox, Detection, PageLayout

    detections_df = pl.read_parquet(detections_file)

    # Filter by page_id or year
    if page_id:
        detections_df = detections_df.filter(pl.col("page_id") == page_id)
    elif year:
        # Filter by year (assuming page_id contains year)
        detections_df = detections_df.filter(pl.col("page_id").str.starts_with(str(year)))
        # Limit pages
        unique_pages = detections_df["page_id"].unique().to_list()[:limit]
        detections_df = detections_df.filter(pl.col("page_id").is_in(unique_pages))
    else:
        click.echo("✗ Must specify either --page-id or --year", err=True)
        return

    if len(detections_df) == 0:
        click.echo(f"✗ No detections found for the specified criteria", err=True)
        return

    # Group by page
    pages_to_visualize = detections_df["page_id"].unique().to_list()
    click.echo(f"✓ Found {len(pages_to_visualize)} page(s) to visualize")

    # Initialize visualizer
    visualizer = LayoutVisualizer(show_text=show_linked_text)

    # Create output directory
    vis_output_dir = config.results_dir / source / "layout" / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # Process element types filter
    element_filter = None if "all" in element_types else list(element_types)

    # Visualize each page
    click.echo(f"\nCreating visualizations...")
    for page_id_to_viz in tqdm(pages_to_visualize, desc="Processing pages"):
        # Get all detections for this page
        page_detections = detections_df.filter(pl.col("page_id") == page_id_to_viz)

        if len(page_detections) == 0:
            continue

        # Get image path (should be same for all detections on page)
        image_path = page_detections["image_path"][0]

        # Reconstruct PageLayout with all detections
        detections = []
        for row in page_detections.iter_rows(named=True):
            det = Detection(
                detection_id=row["detection_id"],
                class_name=row["class_name"],
                confidence=row["confidence"],
                bbox=BoundingBox(
                    x1=row["bbox_x1"],
                    y1=row["bbox_y1"],
                    x2=row["bbox_x2"],
                    y2=row["bbox_y2"],
                ),
                page_id=row["page_id"],
            )
            detections.append(det)

        page_layout = PageLayout(
            page_id=page_id_to_viz,
            image_path=image_path,
            detections=detections,
        )

        # Create visualization
        output_name = f"{page_layout.page_id}_{'comparison' if comparison else 'annotated'}.jpg"
        output_path = vis_output_dir / output_name

        if comparison:
            visualizer.visualize_comparison(page_layout, output_path)
        else:
            visualizer.visualize_page(page_layout, output_path, element_filter)

    click.echo(f"\n{'=' * 60}")
    click.echo("Visualization Complete!")
    click.echo(f"{'=' * 60}")
    click.echo(f"Visualizations saved to: {vis_output_dir}")
    click.echo(f"{'=' * 60}\n")


@layout_group.command()
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--year",
    type=int,
    help="Process only specific year",
)
@click.option(
    "--text-data",
    help="Text data source: 'raw' (default), 'preprocessed', or path to parquet file",
)
def build_articles(source, year, text_data):
    """
    Reconstruct articles from headlines and text blocks.

    Uses matched headlines as anchors to group following text blocks
    into coherent articles.

    Example:
        newspaper-explorer analyze layout build-articles --source der_tag --year 1902
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'=' * 60}")
    click.echo("Article Reconstruction")
    click.echo(f"{'=' * 60}\n")

    config = get_config()

    # Load matched headlines
    headlines_path = config.results_dir / source / "layout" / f"{source}_headlines.parquet"

    if not headlines_path.exists():
        click.echo(f"✗ Matched headlines not found: {headlines_path}", err=True)
        click.echo("  Run 'newspaper-explorer analyze layout match-headlines' first", err=True)
        return

    click.echo("Loading matched headlines...")
    import polars as pl

    headlines_df = pl.read_parquet(headlines_path)

    if year:
        headlines_df = headlines_df.filter(pl.col("year") == year)

    click.echo(f"✓ Loaded {len(headlines_df)} matched headlines")

    # Load text lines
    lines_path = get_text_data_path(source, text_data)
    click.echo(f"Loading text data from: {lines_path}")
    lines_df = pl.read_parquet(lines_path)

    if year:
        lines_df = lines_df.filter(pl.col("year") == year)

    click.echo(f"✓ Loaded {len(lines_df)} text lines")

    # TODO: Load detection results for media (images, tables)
    # For now, we'll build articles with just headlines and text

    # Initialize builder
    builder = ArticleBuilder()

    # Build articles (this needs proper Headline object reconstruction from DataFrame)
    # For now, placeholder
    click.echo("\n⚠ Article building requires full implementation")
    click.echo("  (Headline object reconstruction from DataFrame)")

    click.echo(f"\n{'=' * 60}\n")
    # For now, placeholder
    click.echo("\n⚠ Article building requires full implementation")
    click.echo("  (Headline object reconstruction from DataFrame)")

    click.echo(f"\n{'=' * 60}\n")
