"""
CLI commands for layout analysis.

Provides commands to:
- Detect layout elements (headlines, images, tables)
- Visualize detections for debugging
- Extract images with captions
- Match headlines to OCR text
- Build articles from headlines and text blocks

Usage:
    newspaper-explorer analyze layout detect --source der_tag --year 1902
    newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001
    newspaper-explorer analyze layout extract-images --source der_tag --year 1902
    newspaper-explorer analyze layout match-headlines --source der_tag --year 1902
    newspaper-explorer analyze layout build-articles --source der_tag --year 1902
"""

import logging
import click
from pathlib import Path
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.loading.loader import DataLoader
from newspaper_explorer.analysis.layout.detector import LayoutDetector
from newspaper_explorer.analysis.layout.headline_matcher import HeadlineMatcher
from newspaper_explorer.analysis.layout.article_builder import ArticleBuilder
from newspaper_explorer.analysis.layout.image_extractor import ImageExtractor
from newspaper_explorer.analysis.layout.visualizer import LayoutVisualizer

logger = logging.getLogger(__name__)


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
    help="Device for inference (default: cuda:0)",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for inference (default: 8)",
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
def detect(source, model_size, device, batch_size, conf_threshold, year, limit):
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

    click.echo(f"\n{'='*60}")
    click.echo("Layout Detection with YOLOv11")
    click.echo(f"{'='*60}\n")

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

    # Initialize detector
    click.echo(f"\nInitializing YOLOv11 {model_size} model...")
    detector = LayoutDetector(
        model_size=model_size,
        device=device,
        batch_size=batch_size,
        conf_threshold=conf_threshold,
    )

    # Run detection
    click.echo("\nDetecting layout elements...")
    page_ids = [p.stem for p in image_paths]

    with tqdm(total=len(image_paths), desc="Processing pages") as pbar:
        # Update progress in batches
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_ids = page_ids[i : i + batch_size]

            batch_results = detector.detect_batch(batch_paths, batch_ids)
            results.extend(batch_results)

            pbar.update(len(batch_paths))

    # Save results
    output_dir = config.results_dir / source / "layout"
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    for page_layout in results:
        output_file = output_dir / f"{page_layout.page_id}_layout.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                page_layout.model_dump(mode="json", exclude_none=True),
                f,
                indent=2,
                ensure_ascii=False,
            )

    # Statistics
    total_detections = sum(len(r.detections) for r in results)
    total_headlines = sum(len(r.headlines) for r in results)
    total_images = sum(len(r.images) for r in results)
    total_captions = sum(len(r.captions) for r in results)

    click.echo(f"\n{'='*60}")
    click.echo("Detection Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Pages processed: {len(results)}")
    click.echo(f"Total detections: {total_detections}")
    click.echo(f"  Headlines: {total_headlines}")
    click.echo(f"  Images: {total_images}")
    click.echo(f"  Captions: {total_captions}")
    click.echo(f"\nResults saved to: {output_dir}")
    click.echo(f"{'='*60}\n")


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
def extract_images(source, year, save_crops):
    """
    Extract images from newspaper pages (without caption matching).

    Extracts detected images and saves crops. For caption matching,
    use the 'match-captions' command after extraction.

    Example:
        newspaper-explorer analyze layout extract-images --source der_tag --year 1902
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'='*60}")
    click.echo("Image Extraction")
    click.echo(f"{'='*60}\n")

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
    from newspaper_explorer.analysis.layout.schemas import PageLayout

    detection_files = list(detections_dir.glob("*_layout.json"))
    if year:
        detection_files = [
            f for f in detection_files if f"/{year}/" in str(f) or f"_{year}_" in f.stem
        ]

    if not detection_files:
        click.echo(f"✗ No detection files found", err=True)
        return

    click.echo(f"✓ Found {len(detection_files)} detection files")

    # Initialize extractor (no caption matching)
    extractor = ImageExtractor()

    # Extract images
    output_dir = config.results_dir / source / "layout" / "images"
    all_images = []

    click.echo("\nExtracting images...")
    for det_file in tqdm(detection_files, desc="Processing pages"):
        with open(det_file, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        # Reconstruct PageLayout (simplified)
        # In production, you'd properly deserialize from JSON
        from newspaper_explorer.analysis.layout.schemas import Detection, BoundingBox

        page_layout = PageLayout(
            page_id=page_data["page_id"],
            image_path=page_data["image_path"],
            detections=[],
        )

        # Add images only (no caption matching here)
        for img_data in page_data.get("images", []):
            det = Detection(
                detection_id=img_data["detection_id"],
                class_name=img_data["class_name"],
                confidence=img_data["confidence"],
                bbox=BoundingBox(**img_data["bbox"]),
                page_id=page_data["page_id"],
            )
            page_layout.images.append(det)

        # Extract images (without caption matching)
        images = extractor.extract_images(page_layout, output_dir, save_crops=save_crops)
        all_images.extend(images)

    # Save metadata
    metadata_path = output_dir.parent / f"{source}_images_metadata.parquet"
    extractor.save_image_metadata(all_images, metadata_path, format="parquet")

    click.echo(f"\n{'='*60}")
    click.echo("Image Extraction Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Total images: {len(all_images)}")
    if save_crops:
        click.echo(f"Crops saved to: {output_dir}")
    click.echo(f"Metadata saved to: {metadata_path}")
    click.echo(f"\n💡 To match captions, run:")
    click.echo(f"   newspaper-explorer analyze layout match-captions --source {source}")
    click.echo(f"{'='*60}\n")


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
def match_captions(source, year, caption_position, search_radius, overlap_threshold):
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

    click.echo(f"\n{'='*60}")
    click.echo("Caption Matching")
    click.echo(f"{'='*60}\n")

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
    from newspaper_explorer.analysis.layout.schemas import Detection, BoundingBox, PageLayout

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
    lines_path = config.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"
    if not lines_path.exists():
        click.echo(f"✗ Parsed text not found: {lines_path}", err=True)
        click.echo("  Run 'newspaper-explorer data parse' first", err=True)
        return

    import polars as pl

    lines_df = pl.read_parquet(lines_path)
    if year:
        lines_df = lines_df.filter(pl.col("year") == year)

    click.echo(f"✓ Loaded {len(lines_df)} text lines for caption extraction")

    # Initialize caption matcher
    from newspaper_explorer.analysis.layout.caption_matcher import CaptionMatcher
    from newspaper_explorer.analysis.layout.alto_linker import ALTOLinker

    alto_linker = ALTOLinker(overlap_threshold=overlap_threshold)
    caption_matcher = CaptionMatcher(
        search_radius=search_radius,
        caption_position=caption_position,
    )

    # Match captions
    all_matched = []

    click.echo("\nMatching captions...")
    for det_file in tqdm(detection_files, desc="Processing pages"):
        with open(det_file, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        # Reconstruct PageLayout
        page_layout = PageLayout(
            page_id=page_data["page_id"],
            image_path=page_data["image_path"],
            detections=[],
        )

        # Add images
        for img_data in page_data.get("images", []):
            det = Detection(
                detection_id=img_data["detection_id"],
                class_name=img_data["class_name"],
                confidence=img_data["confidence"],
                bbox=BoundingBox(**img_data["bbox"]),
                page_id=page_data["page_id"],
            )
            page_layout.images.append(det)

        # Add captions
        for cap_data in page_data.get("captions", []):
            det = Detection(
                detection_id=cap_data["detection_id"],
                class_name=cap_data["class_name"],
                confidence=cap_data["confidence"],
                bbox=BoundingBox(**cap_data["bbox"]),
                page_id=page_data["page_id"],
            )
            page_layout.captions.append(det)

        # Extract caption OCR text
        if page_layout.captions:
            page_layout.captions = alto_linker.link_detections_to_text(
                detections=page_layout.captions,
                lines_df=lines_df,
                page_id=page_layout.page_id,
            )

        # Match captions to images
        caption_matches = caption_matcher.match_captions_to_images(
            images=page_layout.images,
            captions=page_layout.captions,
        )

        images_with_captions = caption_matcher.apply_caption_matches(caption_matches)
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
                "caption_id": img.caption_id,
                "caption_confidence": img.caption_confidence,
                "caption_distance": img.caption_distance,
            }
            for img in all_matched
        ]
        import polars as pl

        df = pl.DataFrame(captions_data)
        df.write_parquet(output_path)

    click.echo(f"\n{'='*60}")
    click.echo("Caption Matching Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Total images: {len(all_matched)}")
    click.echo(f"Images with captions: {sum(1 for img in all_matched if img.caption_text)}")
    click.echo(f"Results saved to: {output_path}")
    click.echo(f"{'='*60}\n")


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
def match_headlines(source, year, overlap_threshold):
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

    click.echo(f"\n{'='*60}")
    click.echo("Headline Matching to OCR Text")
    click.echo(f"{'='*60}\n")

    config = get_config()

    # Load parsed lines
    click.echo("Loading parsed text data...")
    lines_path = config.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"

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
        from newspaper_explorer.analysis.layout.schemas import Detection, BoundingBox, PageLayout

        page_layout = PageLayout(
            page_id=page_data["page_id"],
            image_path=page_data["image_path"],
            detections=[],
            year=page_data.get("year", 0),
        )

        for hl_data in page_data.get("headlines", []):
            det = Detection(
                detection_id=hl_data["detection_id"],
                class_name=hl_data["class_name"],
                confidence=hl_data["confidence"],
                bbox=BoundingBox(**hl_data["bbox"]),
                page_id=page_data["page_id"],
            )
            page_layout.headlines.append(det)

        # Match headlines (using DataFrame, not ALTO XML directly)
        headlines = matcher.match_headlines(page_layout, None, lines_df)
        all_headlines.extend(headlines)

    # Save matched headlines
    output_dir = config.results_dir / source / "layout"
    output_path = output_dir / f"{source}_headlines.parquet"

    if all_headlines:
        headlines_data = [hl.model_dump(mode="json", exclude_none=True) for hl in all_headlines]
        import polars as pl

        df = pl.DataFrame(headlines_data)
        df.write_parquet(output_path)

    click.echo(f"\n{'='*60}")
    click.echo("Headline Matching Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Total headlines matched: {len(all_headlines)}")
    click.echo(f"Results saved to: {output_path}")
    click.echo(f"{'='*60}\n")


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
    "--show-text/--no-show-text",
    default=True,
    help="Show matched OCR text on visualizations (default: yes)",
)
def visualize(source, page_id, year, limit, element_types, comparison, show_text):
    """
    Visualize detected layout elements for debugging.
    
    Creates annotated images showing detected regions with bounding boxes,
    labels, confidence scores, and matched OCR text.
    
    Examples:
        # Visualize specific page
        newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001
        
        # Visualize first 10 pages of a year
        newspaper-explorer analyze layout visualize --source der_tag --year 1902 --limit 10
        
        # Show only headlines and images
        newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 \\
            --element-types title --element-types picture
        
        # Create comparison view
        newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 --comparison
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    click.echo(f"\n{'='*60}")
    click.echo("Layout Visualization")
    click.echo(f"{'='*60}\n")

    config = get_config()

    # Load detection results
    detections_dir = config.results_dir / source / "layout"
    if not detections_dir.exists():
        click.echo(f"✗ No layout detections found in {detections_dir}", err=True)
        click.echo("  Run 'newspaper-explorer analyze layout detect' first", err=True)
        return

    # Find detection files to visualize
    import json
    from newspaper_explorer.analysis.layout.schemas import Detection, BoundingBox, PageLayout

    if page_id:
        # Visualize specific page
        detection_files = list(detections_dir.glob(f"{page_id}_layout.json"))
        if not detection_files:
            detection_files = list(detections_dir.glob(f"**/{page_id}_layout.json"))
    elif year:
        # Visualize pages from year
        detection_files = []
        for det_file in detections_dir.rglob("*_layout.json"):
            if f"/{year}/" in str(det_file) or f"_{year}_" in det_file.stem:
                detection_files.append(det_file)
                if len(detection_files) >= limit:
                    break
    else:
        click.echo("✗ Must specify either --page-id or --year", err=True)
        return

    if not detection_files:
        click.echo(f"✗ No detection files found", err=True)
        return

    click.echo(f"✓ Found {len(detection_files)} detection file(s)")

    # Initialize visualizer
    visualizer = LayoutVisualizer(show_text=show_text)

    # Create output directory
    vis_output_dir = config.results_dir / source / "layout" / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # Process element types filter
    element_filter = None if "all" in element_types else list(element_types)

    # Visualize each page
    click.echo(f"\nCreating visualizations...")
    for det_file in tqdm(detection_files, desc="Processing pages"):
        with open(det_file, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        # Reconstruct PageLayout
        page_layout = PageLayout(
            page_id=page_data["page_id"],
            image_path=page_data["image_path"],
            detections=[],
        )

        # Add all detections
        for det_type in ["headlines", "images", "captions", "tables", "text_blocks"]:
            for det_data in page_data.get(det_type, []):
                det = Detection(
                    detection_id=det_data["detection_id"],
                    class_name=det_data["class_name"],
                    confidence=det_data["confidence"],
                    bbox=BoundingBox(**det_data["bbox"]),
                    page_id=page_data["page_id"],
                    text_content=det_data.get("text_content"),
                )
                page_layout.detections.append(det)

                # Organize by type for comparison view
                if "title" in det.class_name.lower() or "header" in det.class_name.lower():
                    page_layout.headlines.append(det)
                elif "picture" in det.class_name.lower():
                    page_layout.images.append(det)
                elif "caption" in det.class_name.lower():
                    page_layout.captions.append(det)
                elif "table" in det.class_name.lower():
                    page_layout.tables.append(det)
                elif "text" in det.class_name.lower():
                    page_layout.text_blocks.append(det)

        # Create visualization
        output_name = f"{page_layout.page_id}_{'comparison' if comparison else 'annotated'}.jpg"
        output_path = vis_output_dir / output_name

        if comparison:
            visualizer.visualize_comparison(page_layout, output_path)
        else:
            visualizer.visualize_page(page_layout, output_path, element_filter)

    click.echo(f"\n{'='*60}")
    click.echo("Visualization Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Visualizations saved to: {vis_output_dir}")
    click.echo(f"{'='*60}\n")


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
def build_articles(source, year):
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

    click.echo(f"\n{'='*60}")
    click.echo("Article Reconstruction")
    click.echo(f"{'='*60}\n")

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
    lines_path = config.data_dir / "raw" / source / "text" / f"{source}_lines.parquet"
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

    click.echo(f"\n{'='*60}\n")
