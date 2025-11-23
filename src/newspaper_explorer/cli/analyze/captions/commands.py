"""Caption enrichment commands for CLI."""

import click
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import polars as pl
from tqdm import tqdm
from PIL import Image

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)


@click.group(name="captions")
def captions_group():
    """Caption extraction and enrichment commands."""
    pass


def calculate_scale_factors(image_path: str, lines_df: pl.DataFrame) -> Tuple[float, float]:
    """
    Calculate scale factors to convert image coordinates to ALTO coordinates.

    ALTO XML coordinates are in a high-resolution reference system that often
    doesn't match the actual image dimensions. Detection bboxes from YOLO are
    in actual image pixel coordinates. We need to scale detection coordinates
    UP to match the ALTO space where text lines are defined.

    Args:
        image_path: Path to the page image
        lines_df: DataFrame of text lines for this page with ALTO coordinates

    Returns:
        Tuple of (scale_x, scale_y) factors to multiply image coordinates by
        to convert them to ALTO space
    """
    try:
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.width, img.height

        # Get max ALTO coordinates from lines
        max_x_val = (lines_df["x"] + lines_df["width"]).max()  # type: ignore
        max_y_val = (lines_df["y"] + lines_df["height"]).max()  # type: ignore

        # Convert to float, handling None
        alto_max_x = float(max_x_val) if max_x_val is not None else 0.0  # type: ignore
        alto_max_y = float(max_y_val) if max_y_val is not None else 0.0  # type: ignore

        # Calculate scale factors: ALTO / image (to scale image coords UP to ALTO space)
        scale_x = alto_max_x / img_width if img_width > 0 else 1.0
        scale_y = alto_max_y / img_height if img_height > 0 else 1.0

        return (float(scale_x), float(scale_y))
    except Exception as e:
        logger.warning(f"Failed to calculate scale factors: {e}")
        return (1.0, 1.0)  # No scaling if we can't determine

    """
    Calculate scale factors to convert image coordinates to ALTO coordinates.

    ALTO XML coordinates are in a high-resolution reference system that often
    doesn't match the actual image dimensions. Detection bboxes from YOLO are
    in actual image pixel coordinates. We need to scale detection coordinates
    UP to match the ALTO space where text lines are defined.

    Args:
        image_path: Path to the page image
        lines_df: DataFrame of text lines for this page with ALTO coordinates

    Returns:
        Tuple of (scale_x, scale_y) factors to multiply image coordinates by
        to convert them to ALTO space
    """
    try:
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.width, img.height

        # Get max ALTO coordinates from lines
        max_x_val = (lines_df["x"] + lines_df["width"]).max()  # type: ignore
        max_y_val = (lines_df["y"] + lines_df["height"]).max()  # type: ignore

        # Convert to float, handling None
        alto_max_x = float(max_x_val) if max_x_val is not None else 0.0  # type: ignore
        alto_max_y = float(max_y_val) if max_y_val is not None else 0.0  # type: ignore

        # Calculate scale factors: ALTO / image (to scale image coords UP to ALTO space)
        scale_x = alto_max_x / img_width if img_width > 0 else 1.0
        scale_y = alto_max_y / img_height if img_height > 0 else 1.0

        return (float(scale_x), float(scale_y))
    except Exception as e:
        logger.warning(f"Failed to calculate scale factors: {e}")
        return (1.0, 1.0)  # No scaling if we can't determine


def find_overlapping_text_in_lines(
    bbox: dict,
    page_lines_df: pl.DataFrame,
    overlap_threshold: float = 0.3,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Optional[str]:
    """
    Find text content that overlaps with a bounding box from parsed lines.

    Note: scale_x and scale_y are used to convert detection bbox coordinates
    (from actual images) to ALTO coordinate space (where lines are defined).

    Args:
        bbox: Dict with bbox_x1, bbox_y1, bbox_x2, bbox_y2 (in image coordinates)
        page_lines_df: DataFrame of text lines for this page (in ALTO coordinates)
        overlap_threshold: Minimum IoU for text overlap
        scale_x: Factor to scale bbox X coordinates to ALTO space
        scale_y: Factor to scale bbox Y coordinates to ALTO space

    Returns:
        Extracted text or None
    """
    if len(page_lines_df) == 0:
        return None

    # Filter lines that have coordinates
    lines_with_coords = page_lines_df.filter(
        (pl.col("x").is_not_null())
        & (pl.col("y").is_not_null())
        & (pl.col("width").is_not_null())
        & (pl.col("height").is_not_null())
    )

    if len(lines_with_coords) == 0:
        return None

    # Calculate line bboxes (already in ALTO space)
    lines_with_bbox = lines_with_coords.with_columns(
        [
            pl.col("x").alias("line_x1"),
            pl.col("y").alias("line_y1"),
            (pl.col("x") + pl.col("width")).alias("line_x2"),
            (pl.col("y") + pl.col("height")).alias("line_y2"),
        ]
    )

    # Scale detection bbox UP to ALTO coordinate space
    bbox_x1, bbox_y1 = bbox["bbox_x1"] * scale_x, bbox["bbox_y1"] * scale_y
    bbox_x2, bbox_y2 = bbox["bbox_x2"] * scale_x, bbox["bbox_y2"] * scale_y
    bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)

    lines_with_iou = (
        lines_with_bbox.with_columns(
            [
                # Intersection coordinates
                pl.max_horizontal(pl.col("line_x1"), pl.lit(bbox_x1)).alias("int_x1"),
                pl.max_horizontal(pl.col("line_y1"), pl.lit(bbox_y1)).alias("int_y1"),
                pl.min_horizontal(pl.col("line_x2"), pl.lit(bbox_x2)).alias("int_x2"),
                pl.min_horizontal(pl.col("line_y2"), pl.lit(bbox_y2)).alias("int_y2"),
            ]
        )
        .with_columns(
            [
                # Intersection dimensions
                (pl.col("int_x2") - pl.col("int_x1")).alias("int_width"),
                (pl.col("int_y2") - pl.col("int_y1")).alias("int_height"),
                # Line area
                (pl.col("width") * pl.col("height")).alias("line_area"),
            ]
        )
        .with_columns(
            [
                # Intersection area (only if positive dimensions)
                pl.when((pl.col("int_width") > 0) & (pl.col("int_height") > 0))
                .then(pl.col("int_width") * pl.col("int_height"))
                .otherwise(0)
                .alias("int_area"),
            ]
        )
        .with_columns(
            [
                # IoU
                (pl.col("int_area") / (pl.col("line_area") + bbox_area - pl.col("int_area"))).alias(
                    "iou"
                )
            ]
        )
    )

    # Filter by threshold and sort by vertical position
    overlapping = lines_with_iou.filter(pl.col("iou") >= overlap_threshold).sort("y")

    if len(overlapping) == 0:
        return None

    # Join text with spaces
    return " ".join(overlapping["text"].to_list())


@captions_group.command(name="enrich")
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--run-id",
    required=True,
    help="Layout detection run ID",
)
@click.option(
    "--overlap-threshold",
    default=0.01,
    type=float,
    help="Minimum IoU for text overlap (0.0-1.0, default: 0.01 for liberal matching)",
)
@click.option(
    "--classes",
    default="Caption,Picture",
    help="Comma-separated list of class names to enrich (default: Caption,Picture)",
)
def enrich_captions(source: str, run_id: str, overlap_threshold: float, classes: str):
    """
    Enrich layout detections with text content from parsed lines.

    Uses proper coordinate scaling via image index for accurate text extraction.

    This command:
    1. Loads layout detections (captions, pictures, etc.)
    2. Loads parsed text lines from lines.parquet
    3. Uses TextLinker with image index for proper coordinate scaling
    4. Extracts overlapping text content with IoU-based matching
    5. Saves enriched parquet with text_content column

    Example:
        newspaper-explorer analyze captions enrich --source der_tag --run-id yolo11m_doclaynet_20251110_010200
    """
    from newspaper_explorer.analyze.layout.text_linker import TextLinker
    from newspaper_explorer.models.analysis.layout import Detection, BoundingBox
    import cv2

    config = get_config()

    # Parse class names
    class_list = [c.strip() for c in classes.split(",")]

    # Load layout detections
    layout_path = Path(config.results_dir) / source / "layout" / run_id / "layout.parquet"

    if not layout_path.exists():
        click.echo(f"❌ Layout file not found: {layout_path}", err=True)
        return

    click.echo(f"📂 Loading layout detections from {layout_path}")
    detections_df = pl.read_parquet(layout_path)

    # Filter by class
    detections_df = detections_df.filter(pl.col("class_name").is_in(class_list))

    click.echo(f"📊 Found {len(detections_df)} detections for classes: {', '.join(class_list)}")

    if len(detections_df) == 0:
        click.echo("⚠️  No detections to process", err=True)
        return

    # Load parsed text lines
    lines_path = Path(config.data_dir) / "raw" / source / "text" / f"{source}_lines.parquet"

    if not lines_path.exists():
        click.echo(f"❌ Lines file not found: {lines_path}", err=True)
        click.echo(f"💡 Run: newspaper-explorer data parse --source {source}", err=True)
        return

    click.echo(f"📂 Loading parsed text lines from {lines_path}")
    lines_df = pl.read_parquet(lines_path)

    # Get unique page_ids from detections
    page_ids = detections_df["page_id"].unique().to_list()
    click.echo(f"📄 Processing {len(page_ids)} unique pages")

    # Filter lines to only pages we need
    lines_df = lines_df.filter(pl.col("page_id").is_in(page_ids))

    # Initialize TextLinker with source name for proper coordinate scaling
    click.echo(f"🔧 Initializing TextLinker with image index for {source}...")
    text_linker = TextLinker(overlap_threshold=overlap_threshold, source_name=source)

    if text_linker.image_index is None:
        click.echo("⚠️  Warning: Image index not loaded. Coordinate scaling may be inaccurate.")

    click.echo(f"🔍 Extracting text content (overlap threshold: {overlap_threshold})...")

    # Process detections page by page for efficiency
    enriched_rows = []

    for page_id in tqdm(page_ids, desc="Processing pages"):
        # Get detections for this page
        page_detections = detections_df.filter(pl.col("page_id") == page_id)
        page_lines = lines_df.filter(pl.col("page_id") == page_id)

        if len(page_lines) == 0:
            # No lines on this page, add detections without text
            for row in page_detections.iter_rows(named=True):
                enriched_rows.append({**row, "text_content": None})
            continue

        # Convert dataframe rows to Detection objects
        detection_objects = []
        row_data_map = {}  # Keep original row data for preserving columns
        for row in page_detections.iter_rows(named=True):
            det_id = row["detection_id"]
            row_data_map[det_id] = row  # Store original row
            detection_objects.append(
                Detection(
                    detection_id=det_id,
                    class_name=row["class_name"],
                    confidence=row["confidence"],
                    bbox=BoundingBox(
                        x1=row["bbox_x1"],
                        y1=row["bbox_y1"],
                        x2=row["bbox_x2"],
                        y2=row["bbox_y2"],
                    ),
                    page_id=page_id,
                    image_path=row.get("image_path"),
                )
            )

        # Get layout dimensions for this page
        layout_width, layout_height = None, None
        if detection_objects and detection_objects[0].image_path:
            image_path = detection_objects[0].image_path
            if Path(image_path).exists():
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        layout_height, layout_width = img.shape[:2]
                except Exception as e:
                    logger.warning(f"Could not load image {image_path}: {e}")

        # Link detections to text with proper coordinate scaling
        linked_detections = text_linker.link_detections_to_text(
            detections=detection_objects,
            lines_df=page_lines,
            page_id=page_id,
            layout_width=layout_width,
            layout_height=layout_height,
        )

        # Convert back to dict format, preserving all original columns
        for det in linked_detections:
            original_row = row_data_map[det.detection_id]
            enriched_row = {**original_row}  # Start with all original columns
            enriched_row["text_content"] = det.text_content  # Add/update text content
            enriched_rows.append(enriched_row)

    # Create enriched dataframe with schema inference from all rows
    enriched_df = pl.DataFrame(enriched_rows, infer_schema_length=None)

    # Count successful extractions
    with_text = enriched_df.filter(pl.col("text_content").is_not_null())
    click.echo(
        f"✅ Extracted text for {len(with_text)} / {len(enriched_df)} detections ({len(with_text) / len(enriched_df) * 100:.1f}%)"
    )

    # Save enriched parquet
    output_path = layout_path.parent / "layout_enriched.parquet"
    enriched_df.write_parquet(output_path)

    click.echo(f"💾 Saved enriched data to {output_path}")

    # Update metadata
    metadata_path = layout_path.parent / "layout.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        from datetime import datetime

        metadata["enriched"] = {
            "created_at": datetime.now().isoformat(),
            "enriched_classes": class_list,
            "overlap_threshold": overlap_threshold,
            "total_detections": len(enriched_df),
            "with_text_content": len(with_text),
            "lines_source": str(lines_path),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        click.echo(f"📝 Updated metadata")


@captions_group.command(name="match")
@click.option(
    "--source",
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--run-id",
    required=True,
    help="Layout detection run ID",
)
@click.option(
    "--max-distance",
    default=200,
    type=int,
    help="Maximum center-to-center distance in pixels to match caption (default: 200)",
)
def match_captions_to_pictures(source: str, run_id: str, max_distance: int):
    """
    Match picture detections with their nearest caption detections.

    This is a two-step process:
    1. First run 'enrich' to extract text from caption detections
    2. Then run 'match' to link pictures with their nearest captions

    Matching algorithm:
    Uses ProximityMatcher to find the best caption for each picture based on:
    - Spatial position (prioritizes below/above)
    - Distance score (vertical distance + horizontal alignment)

    This command:
    1. Loads enriched layout data (with caption text extracted)
    2. For each picture, finds the nearest caption on the same page
    3. Copies caption text AND structural link (ID, bbox) to picture
    4. Saves updated enriched parquet

    Example:
        # Step 1: Extract text for captions
        newspaper-explorer analyze captions enrich --source der_tag --run-id yolo11m_doclaynet_20251110_010200 --classes Caption

        # Step 2: Match pictures with captions
        newspaper-explorer analyze captions match --source der_tag --run-id yolo11m_doclaynet_20251110_010200
    """
    from newspaper_explorer.analyze.layout.region_matching import ProximityMatcher
    from newspaper_explorer.models.analysis.layout import Detection, BoundingBox

    config = get_config()

    # Load original layout data for pictures
    layout_path = Path(config.results_dir) / source / "layout" / run_id / "layout.parquet"
    enriched_path = (
        Path(config.results_dir) / source / "layout" / run_id / "layout_enriched.parquet"
    )

    if not layout_path.exists():
        click.echo(f"❌ Layout file not found: {layout_path}", err=True)
        return

    if not enriched_path.exists():
        click.echo(f"❌ Enriched layout file not found: {enriched_path}", err=True)
        click.echo(
            "💡 Run 'enrich --classes Caption' command first to extract caption text", err=True
        )
        return

    click.echo(f"📂 Loading layout data from {layout_path}")
    all_detections_df = pl.read_parquet(layout_path)

    click.echo(f"📂 Loading enriched captions from {enriched_path}")
    enriched_captions_df = pl.read_parquet(enriched_path)

    # Get pictures from original layout
    pictures_df = all_detections_df.filter(pl.col("class_name") == "Picture")
    captions_df = enriched_captions_df.filter(pl.col("class_name") == "Caption")

    click.echo(f"📊 Found {len(pictures_df)} pictures and {len(captions_df)} captions")

    captions_with_text = captions_df.filter(pl.col("text_content").is_not_null())
    click.echo(f"📝 {len(captions_with_text)} captions have extracted text")

    if len(captions_with_text) == 0:
        click.echo("❌ No captions with text found. Run 'enrich --classes Caption' first", err=True)
        return

    # Match pictures to captions page-by-page using spatial-aware matching
    click.echo(f"🔍 Matching pictures with nearest captions (max distance: {max_distance}px)...")
    click.echo("📐 Using ProximityMatcher: prioritizes below/above position over pure distance")

    # Initialize matcher
    matcher = ProximityMatcher(search_radius=max_distance, relative_position="any")

    page_ids = pictures_df["page_id"].unique().to_list()
    enriched_pictures = []
    matched_count = 0

    for page_id in tqdm(page_ids, desc="Processing pages"):
        # Get pictures and captions for this page only
        page_pictures = pictures_df.filter(pl.col("page_id") == page_id)
        page_captions = captions_with_text.filter(pl.col("page_id") == page_id)

        if len(page_captions) == 0:
            # No captions on this page, keep pictures as is
            for row in page_pictures.iter_rows(named=True):
                enriched_pictures.append(
                    {**row, "text_content": None, "caption_id": None, "caption_bbox": None}
                )
            continue

        # Convert to Detection objects for matcher
        source_dets = []
        for row in page_pictures.iter_rows(named=True):
            source_dets.append(
                Detection(
                    detection_id=row["detection_id"],
                    class_name=row["class_name"],
                    confidence=row["confidence"],
                    bbox=BoundingBox(
                        x1=row["bbox_x1"],
                        y1=row["bbox_y1"],
                        x2=row["bbox_x2"],
                        y2=row["bbox_y2"],
                    ),
                    page_id=page_id,
                    image_path=row.get("image_path"),
                )
            )

        target_dets = []
        for row in page_captions.iter_rows(named=True):
            target_dets.append(
                Detection(
                    detection_id=row["detection_id"],
                    class_name=row["class_name"],
                    confidence=row["confidence"],
                    bbox=BoundingBox(
                        x1=row["bbox_x1"],
                        y1=row["bbox_y1"],
                        x2=row["bbox_x2"],
                        y2=row["bbox_y2"],
                    ),
                    page_id=page_id,
                    image_path=row.get("image_path"),
                    text_content=row.get("text_content"),
                )
            )

        # Run matching
        # We don't need to extract text again, so pass extract_text=False
        matches = matcher.match_elements(
            source_elements=source_dets, target_elements=target_dets, extract_text=False
        )

        # Process matches
        for src, tgt in matches:
            # Find original row to preserve all columns
            original_row = page_pictures.filter(pl.col("detection_id") == src.detection_id).row(
                0, named=True
            )

            enriched_row = {**original_row}

            if tgt:
                matched_count += 1
                enriched_row["text_content"] = tgt.text_content
                enriched_row["caption_id"] = tgt.detection_id
                # Store bbox as struct or JSON string? Parquet supports structs.
                # For simplicity and compatibility with Polars, we might want to flatten or keep as struct.
                # Let's try to store as a struct if possible, or just individual columns if needed.
                # But wait, the backend expects a dict for bbox.
                # Let's store it as a struct in the dataframe.
                enriched_row["caption_bbox"] = {
                    "x1": tgt.bbox.x1,
                    "y1": tgt.bbox.y1,
                    "x2": tgt.bbox.x2,
                    "y2": tgt.bbox.y2,
                }
            else:
                enriched_row["text_content"] = None
                enriched_row["caption_id"] = None
                enriched_row["caption_bbox"] = None

            enriched_pictures.append(enriched_row)

    # Combine matched pictures with enriched captions and other detections
    enriched_pictures_df = pl.DataFrame(enriched_pictures, infer_schema_length=None)

    # Concat pictures with captions (both already have text_content, caption_id, caption_bbox)
    # Then separately handle other detections
    pictures_and_captions = pl.concat([enriched_captions_df, enriched_pictures_df], how="diagonal")

    # Get other detections (not pictures or captions) from original layout
    other_detections_df = all_detections_df.filter(
        ~pl.col("class_name").is_in(["Picture", "Caption"])
    )

    # Concat all with diagonal mode (handles missing columns automatically)
    enriched_df = pl.concat([pictures_and_captions, other_detections_df], how="diagonal")

    click.echo(
        f"✅ Matched {matched_count} / {len(pictures_df)} pictures with captions ({matched_count / len(pictures_df) * 100:.1f}%)"
    )

    # Save enriched data
    enriched_df.write_parquet(enriched_path)
    click.echo(f"💾 Saved updated enriched data to {enriched_path}")

    # Update metadata
    metadata_path = enriched_path.parent / "layout.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        from datetime import datetime

        metadata["caption_matching"] = {
            "matched_at": datetime.now().isoformat(),
            "max_distance": max_distance,
            "pictures_matched": matched_count,
            "total_pictures": len(pictures_df),
            "match_rate": matched_count / len(pictures_df) if len(pictures_df) > 0 else 0,
            "method": "ProximityMatcher",
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        click.echo(f"📝 Updated metadata")


if __name__ == "__main__":
    captions_group()
