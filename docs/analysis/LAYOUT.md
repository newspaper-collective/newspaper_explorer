# Layout Analysis Documentation

Complete guide to the YOLOv11-based layout analysis system for historical newspaper processing.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [CLI Commands](#cli-commands)
5. [Python API](#python-api)
6. [Detected Element Types](#detected-element-types)
7. [Advanced Features](#advanced-features)
8. [Performance & Configuration](#performance--configuration)
9. [Data Schemas](#data-schemas)
10. [Complete Workflows](#complete-workflows)

---

## Overview

The layout analysis system provides comprehensive document structure detection and content extraction for historical newspapers:

**Core Capabilities**:
1. **Detection** - YOLOv11-based detection of 11 document element types
2. **Text Matching** - Unified OCR text extraction for any detected element
3. **Region Extraction** - Extract and crop any detected element type (images, headlines, etc.)
4. **Proximity Matching** - Match related elements based on spatial relationships
5. **Visualization** - Debug visualizations of detected regions
6. **Filtering** - Coordinate-based filtering to exclude false positives

**Key Features**:
- ✅ **Generalized API**: Works with any element type, not just images
- ✅ **DataFrame-based**: Fast batch processing using pre-parsed Polars DataFrames
- ✅ **Coordinate filtering**: Exclude headers, footers, and small decorative elements
- ✅ **Spatial reasoning**: Match elements by proximity (above, below, left, right)
- ✅ **Library-first design**: No CLI mixing, proper logging, type hints

---

## Architecture

### File Structure

```
src/newspaper_explorer/analysis/layout/
├── detector.py              # YOLOv11 wrapper (library-style, no CLI)
├── text_linker.py           # Universal text extraction (DataFrame-based, FAST)
├── headline_matcher.py      # Headline-specific wrapper using TextLinker
├── region_extraction.py     # RegionExtractor (generalized from ImageExtractor)
├── caption_matching.py      # ProximityMatcher (generalized from CaptionMatcher)
├── article_builder.py       # Article reconstruction
├── visualizer.py            # Debug visualizations
└── schemas.py               # Data schemas (Detection, Headline, Article)

cli/analyze/layout/
└── commands.py              # Layout analysis CLI commands
```

### Integration with Data Pipeline

```
Download → Parse ALTO/METS → Polars DataFrame
                                    ↓
                            Layout Detection (YOLOv11)
                                    ↓
                          Text Matching (ALTO coordinates)
                                    ↓
                    Region Extraction + Proximity Matching
                                    ↓
                       Headlines/Images/Articles + Visualization
```

**Integration Points**:
- Uses existing Polars DataFrames from `DataLoader`
- Reads ALTO XML coordinates directly (no re-parsing)
- Outputs to standard `results/{source}/layout/` directory
- Follows configuration-driven patterns

---

## Core Components

### 1. LayoutDetector

**Purpose**: Detect document layout elements using YOLOv11.

**Features**:
- Three model sizes: nano (fast), small (balanced), medium (accurate)
- Batch processing with GPU support
- Progress tracking
- Automatic model download from HuggingFace

**Usage**:
```python
from newspaper_explorer.analysis.layout.detector import LayoutDetector

detector = LayoutDetector(model_size="small", batch_size=8)

# Detect from image path
detections = detector.detect_from_image("path/to/page.jpg")

# Batch detection from multiple pages
pages = ["page1.jpg", "page2.jpg", "page3.jpg"]
all_detections = detector.detect_batch(pages)
```

### 2. TextLinker

**Purpose**: Universal OCR text extraction for any detected element using bounding box overlap.

**Why DataFrame-based?**: Process thousands of pages without re-parsing XML files.

**Usage**:
```python
from newspaper_explorer.analysis.layout.text_linker import TextLinker
from newspaper_explorer.data.loading.loader import DataLoader

# Load parquet ONCE (contains ALL pages)
df = DataLoader.load_parquet("data/processed/der_tag/text/der_tag_lines.parquet")

linker = TextLinker(overlap_threshold=0.3)

# Process many pages - NO XML parsing overhead!
for page_id in page_ids:
    matched = linker.link_detections_to_text(
        detections=detections,
        lines_df=df,  # Reuse same DataFrame
        page_id=page_id
    )
```

### 3. RegionExtractor (formerly ImageExtractor)

**Purpose**: Extract and crop **any** type of detected region from page images.

**Generalization**: Not just images - works with headlines, text blocks, tables, etc.

**Features**:
- Region type filtering
- Coordinate-based filtering (exclude headers/footers)
- Size-based filtering (min width/height)
- Padding control
- Metadata export (Parquet/JSON)

**Basic Usage**:
```python
from newspaper_explorer.analysis.layout.region_extraction import RegionExtractor

extractor = RegionExtractor(padding=5)

# Extract images
images = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type="picture",  # Filter by type
)

# Extract headlines
headlines = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type="title",
)

# Extract multiple types
visual_elements = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type=["picture", "figure", "chart"],  # Can pass list
)

# Extract all regions
all_regions = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type=None,  # No type filtering
)
```

**With Coordinate Filtering**:
```python
# Exclude newspaper headers and footers
extractor = RegionExtractor(
    padding=5,
    exclude_top_percent=15.0,      # Exclude top 15% (headers)
    exclude_bottom_percent=5.0,    # Exclude bottom 5% (footers)
    min_region_height=100,         # Min 100px height
    min_region_width=100,          # Min 100px width
)

extracted = extractor.extract_regions(
    detections=detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type="picture",
)

# Save metadata
extractor.save_region_metadata(
    detections=extracted,
    output_path="output.parquet",
)
```

### 4. ProximityMatcher (formerly CaptionMatcher)

**Purpose**: Match **any** element type to **any** other element type based on spatial proximity.

**Generalization**: Not just captions - match bylines to headlines, sidebars to articles, etc.

**Features**:
- Five spatial modes: `above`, `below`, `left`, `right`, `any`
- Configurable search radius
- Overlap threshold
- OCR text extraction for matched elements

**Usage Examples**:

```python
from newspaper_explorer.analysis.layout.caption_matching import ProximityMatcher

# Match captions to images (classic use case)
matcher = ProximityMatcher(
    search_radius=150,
    relative_position="below",  # Caption below image
    overlap_threshold=0.3,
)

matches = matcher.match_elements(
    source_elements=images,
    target_elements=captions,
    lines_df=lines_df,
    page_id=page_id,
    extract_text=True,  # Extract OCR text for captions
)

images_with_captions = matcher.apply_matches(matches, target_attr="caption")

# Match bylines to headlines
matcher = ProximityMatcher(
    search_radius=100,
    relative_position="below",  # Byline below headline
)

matches = matcher.match_elements(
    source_elements=headlines,
    target_elements=bylines,
    extract_text=True,
)

# Match sidebar to main article
matcher = ProximityMatcher(
    search_radius=200,
    relative_position="right",  # Sidebar to the right
)

matches = matcher.match_elements(
    source_elements=main_articles,
    target_elements=sidebars,
)
```

### 5. LayoutVisualizer

**Purpose**: Debug and verify detected regions with annotated visualizations.

**Features**:
- Color-coded bounding boxes by element type
- Confidence scores
- Matched OCR text overlay
- Legend with element counts
- Comparison mode (side-by-side panels)

**Usage**:
```python
from newspaper_explorer.analysis.layout.visualizer import LayoutVisualizer

visualizer = LayoutVisualizer(show_text=True, show_confidence=True)

# Single view with all elements
visualizer.visualize_page(page_layout, "output.jpg")

# Comparison view (separate panels for each element type)
visualizer.visualize_comparison(page_layout, "comparison.jpg")

# Quick visualization
from newspaper_explorer.analysis.layout.visualizer import quick_visualize
quick_visualize(page_layout, "debug.jpg", element_types=["title", "picture"])
```

---

## CLI Commands

All commands follow: `newspaper-explorer analyze layout <command> --source <name>`

### 1. Detect Layout Elements

```bash
# Detect all layout elements in a year
newspaper-explorer analyze layout detect --source der_tag --year 1902

# Use different model size (nano/small/medium)
newspaper-explorer analyze layout detect --source der_tag --model-size nano

# Limit processing for testing
newspaper-explorer analyze layout detect --source der_tag --year 1902 --limit 10
```

**Output**: JSON files in `results/der_tag/layout/detections/` with detected elements

### 2. Visualize Detections

```bash
# Visualize specific page
newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001

# Visualize first 10 pages of a year
newspaper-explorer analyze layout visualize --source der_tag --year 1902 --limit 10

# Show only headlines and images
newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 \
    --element-types title --element-types picture

# Create comparison view
newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 --comparison

# Without OCR text overlay
newspaper-explorer analyze layout visualize --source der_tag --page-id 1902_01_01_001 --no-show-text
```

**Output**: Annotated images in `results/der_tag/layout/visualizations/`

### 3. Extract Pictures with Captions

```bash
# Basic extraction
newspaper-explorer analyze layout extract-pictures --source der_tag --year 1902

# With coordinate filtering (exclude headers/footers)
newspaper-explorer analyze layout extract-pictures \
    --source der_tag \
    --year 1902 \
    --exclude-top-percent 15 \
    --exclude-bottom-percent 5

# With size filtering
newspaper-explorer analyze layout extract-pictures \
    --source der_tag \
    --year 1902 \
    --min-height 100 \
    --min-width 100

# Caption position options
newspaper-explorer analyze layout extract-pictures \
    --source der_tag \
    --year 1902 \
    --caption-position below  # or: above, both

# Skip saving image crops (metadata only)
newspaper-explorer analyze layout extract-pictures \
    --source der_tag \
    --year 1902 \
    --no-save-crops
```

**Output**: 
- Cropped images: `results/der_tag/layout/pictures/`
- Metadata: `results/der_tag/layout/der_tag_pictures_metadata.parquet`

### 4. Match Headlines to OCR

```bash
# Match detected headlines to OCR text from ALTO XML
newspaper-explorer analyze layout match-headlines --source der_tag --year 1902

# Adjust overlap threshold
newspaper-explorer analyze layout match-headlines \
    --source der_tag \
    --year 1902 \
    --overlap-threshold 0.5
```

**Output**: `results/der_tag/layout/der_tag_headlines.parquet`

### 5. Build Articles (TODO)

```bash
# Reconstruct articles from headlines and text blocks
newspaper-explorer analyze layout build-articles --source der_tag --year 1902
```

*Note: This command needs full implementation (headline object reconstruction from DataFrame).*

---

## Python API

### Complete Example: Process a Year

```python
from newspaper_explorer.analysis.layout.detector import LayoutDetector
from newspaper_explorer.analysis.layout.text_linker import TextLinker
from newspaper_explorer.analysis.layout.region_extraction import RegionExtractor
from newspaper_explorer.analysis.layout.caption_matching import ProximityMatcher
from newspaper_explorer.analysis.layout.visualizer import LayoutVisualizer
from newspaper_explorer.data.loading.loader import DataLoader
from pathlib import Path

# 1. Setup
source = "der_tag"
year = 1902
results_dir = Path(f"results/{source}/layout")
results_dir.mkdir(parents=True, exist_ok=True)

# 2. Load OCR data ONCE
df = DataLoader.load_parquet(f"data/processed/{source}/text/{source}_lines.parquet")

# 3. Initialize components
detector = LayoutDetector(model_size="small")
linker = TextLinker(overlap_threshold=0.3)
extractor = RegionExtractor(
    padding=5,
    exclude_top_percent=15.0,  # Filter headers
    exclude_bottom_percent=5.0,  # Filter footers
)
matcher = ProximityMatcher(
    search_radius=150,
    relative_position="below",
)
visualizer = LayoutVisualizer(show_text=True)

# 4. Process pages
image_dir = Path(f"data/raw/{source}/images/{year}")
for image_path in image_dir.glob("*.jpg"):
    page_id = image_path.stem
    
    # Detect layout
    page_layout = detector.detect_from_image(str(image_path))
    
    # Link to OCR text
    linker.link_detections_to_text(
        detections=page_layout.all_detections,
        lines_df=df,
        page_id=page_id
    )
    
    # Extract pictures
    pictures = extractor.extract_regions(
        detections=page_layout.all_detections,
        page_layout=page_layout,
        output_dir=results_dir / "pictures",
        region_type="picture",
    )
    
    # Match captions to pictures
    captions = [d for d in page_layout.all_detections if d.class_name == "caption"]
    matches = matcher.match_elements(
        source_elements=pictures,
        target_elements=captions,
        lines_df=df,
        page_id=page_id,
        extract_text=True,
    )
    pictures_with_captions = matcher.apply_matches(matches, target_attr="caption")
    
    # Visualize for debugging
    visualizer.visualize_page(
        page_layout,
        str(results_dir / "visualizations" / f"{page_id}.jpg")
    )

# 5. Save metadata
extractor.save_region_metadata(
    detections=pictures_with_captions,
    output_path=results_dir / f"{source}_pictures_metadata.parquet",
)
```

---

## Detected Element Types

YOLOv11 model detects 11 DocLayNet categories:

| Element | Description | Use Case |
|---------|-------------|----------|
| **title** | Main article headlines | Article reconstruction |
| **section-header** | Sub-headlines | Article structure |
| **picture** | Images/photos | Visual content extraction |
| **caption** | Image captions | Image metadata |
| **table** | Tables | Structured data extraction |
| **text** | Body text paragraphs | Article content |
| **list-item** | List elements | Structured content |
| **formula** | Mathematical formulas | Scientific content |
| **page-header** | Page headers | Layout understanding |
| **page-footer** | Page footers | Layout understanding |
| **footnote** | Footnotes | Supplementary content |

---

## Advanced Features

### Coordinate-Based Filtering

**Problem**: Layout detection models often misclassify newspaper headers (masthead, logo) as pictures, leading to many duplicate extractions.

**Solution**: Filter regions based on position and size.

#### Filtering Parameters

- **`exclude_top_percent`**: Exclude regions in the top X% of the page (e.g., 15 for headers)
- **`exclude_bottom_percent`**: Exclude regions in the bottom X% of the page (e.g., 5 for footers)
- **`min_region_height`**: Minimum height in pixels for regions to extract
- **`min_region_width`**: Minimum width in pixels for regions to extract

#### How Filtering Works

1. Load page image to get dimensions (height, width)
2. Apply filters in order:
   - Region type filter (if specified)
   - Top exclusion zone (headers)
   - Bottom exclusion zone (footers)
   - Minimum height constraint
   - Minimum width constraint
3. Extract and crop remaining regions
4. Log statistics about excluded regions

#### Finding Optimal Values

Start with these typical values and adjust:

- **Headers**: 10-20% from top (varies by newspaper design)
- **Footers**: 3-5% from bottom (page numbers)
- **Minimum size**: 50-100px (filter out small decorative elements)

**Quick test**:
```bash
# Test different top exclusion values
for percent in 5 10 15 20; do
    newspaper-explorer analyze layout extract-pictures \
        --source der_tag \
        --year 1902 \
        --exclude-top-percent $percent \
        --limit 10
done
```

#### Filtering Logs

The extractor logs detailed information:

```
INFO: Filtered 3/15 regions: top=2, bottom=1, size=0
```

Enable debug logging for individual exclusions:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Spatial Proximity Matching

**Purpose**: Match related elements based on their spatial relationship.

**Supported positions**:
- `above` - Target is above source
- `below` - Target is below source
- `left` - Target is to the left of source
- `right` - Target is to the right of source
- `any` - Match closest target regardless of position

**Key parameters**:
- `search_radius`: Maximum distance in pixels
- `overlap_threshold`: Minimum horizontal/vertical alignment (0-1)

**Use cases**:
1. Match captions to images (below)
2. Match bylines to headlines (below)
3. Match sidebars to articles (left/right)
4. Match footnotes to references (any)

---

## Performance & Configuration

### Model Sizes

| Size | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| **nano** | Fastest | Good | Quick testing, large batches |
| **small** | Fast | Better | Production (balanced) |
| **medium** | Slower | Best | High-accuracy requirements |

### Batch Processing

- Default batch size: 8 images
- GPU recommended (CUDA)
- Can process on CPU (slower)
- Progress tracking with tqdm

### Memory Requirements

- Layout detection: ~2-4GB GPU memory
- Text matching: Minimal overhead
- Visualization: Load full page images (can be large)

### Performance Tips

1. **Use DataFrame-based text linking**: Load parquet once, reuse for all pages
2. **Batch detection**: Process multiple pages in one call
3. **GPU acceleration**: Use CUDA-capable GPU for faster inference
4. **Appropriate model size**: Use nano for large batches, medium for quality
5. **Coordinate filtering**: Filter regions before cropping to save time

---

## Data Schemas

### Detection
```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Detection:
    detection_id: str
    class_name: str          # Element type (title, picture, caption, etc.)
    confidence: float        # Detection confidence (0-1)
    bbox: BoundingBox       # Bounding box coordinates
    page_id: str
    
    # Matched from ALTO XML
    text_content: Optional[str] = None
    alto_elements: List[str] = field(default_factory=list)
    
    # For region extraction
    image_path: Optional[str] = None
    
    # For proximity matching
    caption: Optional["Detection"] = None
    caption_text: Optional[str] = None
```

### BoundingBox
```python
@dataclass
class BoundingBox:
    x: int      # Top-left x coordinate
    y: int      # Top-left y coordinate
    width: int  # Box width
    height: int # Box height
```

### PageLayout
```python
@dataclass
class PageLayout:
    page_id: str
    image_path: str
    all_detections: List[Detection]
    
    # Access by element type
    def get_by_type(self, class_name: str) -> List[Detection]:
        return [d for d in self.all_detections if d.class_name == class_name]
```

### Headline
```python
@dataclass
class Headline:
    headline_id: str
    detection: Detection
    ocr_text: str           # Matched text from OCR
    text_block_ids: List[str]
    confidence: float
    match_score: float
    page_id: str
    year: int
    date: Optional[datetime] = None
```

### Article (TODO)
```python
@dataclass
class Article:
    article_id: str
    headline: Headline
    text_blocks: List[str]  # Text block IDs
    full_text: str
    page_id: str
    year: int
    
    # Associated media
    images: List[Detection] = field(default_factory=list)
    tables: List[Detection] = field(default_factory=list)
```

---

## Complete Workflows

### Workflow 1: Extract Pictures with Captions

```bash
# 1. Detect layout elements
newspaper-explorer analyze layout detect \
    --source der_tag \
    --year 1902 \
    --model-size small

# 2. Visualize sample pages to verify detection quality
newspaper-explorer analyze layout visualize \
    --source der_tag \
    --year 1902 \
    --limit 5 \
    --comparison

# 3. Extract pictures with caption matching and filtering
newspaper-explorer analyze layout extract-pictures \
    --source der_tag \
    --year 1902 \
    --exclude-top-percent 15 \
    --exclude-bottom-percent 5 \
    --min-height 100 \
    --min-width 100

# 4. Check outputs
ls -lh results/der_tag/layout/pictures/
head results/der_tag/layout/der_tag_pictures_metadata.parquet
```

### Workflow 2: Headline Extraction

```bash
# 1. Detect layout
newspaper-explorer analyze layout detect \
    --source der_tag \
    --year 1902

# 2. Match headlines to OCR
newspaper-explorer analyze layout match-headlines \
    --source der_tag \
    --year 1902 \
    --overlap-threshold 0.3

# 3. Analyze results
head results/der_tag/layout/der_tag_headlines.parquet
```

### Workflow 3: Debug Detection Quality

```bash
# 1. Run on small sample
newspaper-explorer analyze layout detect \
    --source der_tag \
    --year 1902 \
    --limit 10

# 2. Visualize with comparison mode
newspaper-explorer analyze layout visualize \
    --source der_tag \
    --year 1902 \
    --limit 10 \
    --comparison

# 3. Check specific element types
newspaper-explorer analyze layout visualize \
    --source der_tag \
    --page-id 1902_01_15_003 \
    --element-types title \
    --element-types picture \
    --element-types caption

# 4. Review outputs
open results/der_tag/layout/visualizations/
```

---

## Migration Guide

### RegionExtractor (formerly ImageExtractor)

**Old Code**:
```python
from newspaper_explorer.analyze.layout.image_extractor import ImageExtractor

extractor = ImageExtractor(padding=5)
images = extractor.extract_images(images, page_layout, output_dir)
extractor.save_image_metadata(images, output_path)
```

**New Code**:
```python
from newspaper_explorer.analyze.layout.region_extraction import RegionExtractor

extractor = RegionExtractor(padding=5)
images = extractor.extract_regions(
    detections=images,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type="picture"
)
extractor.save_region_metadata(images, output_path)
```

### ProximityMatcher (formerly CaptionMatcher)

**Old Code**:
```python
from newspaper_explorer.analyze.layout.caption_matching import CaptionMatcher

matcher = CaptionMatcher(
    search_radius=150,
    caption_position="below",  # Old param name
)
images_with_captions = matcher.match_captions_for_page(page_layout, lines_df)
```

**New Code**:
```python
from newspaper_explorer.analyze.layout.caption_matching import ProximityMatcher

matcher = ProximityMatcher(
    search_radius=150,
    relative_position="below",  # New param name
)

# Explicit workflow
matches = matcher.match_elements(
    source_elements=images,
    target_elements=captions,
    lines_df=lines_df,
    page_id=page_id,
    extract_text=True
)
images_with_captions = matcher.apply_matches(matches, target_attr="caption")
```

**No Backward Compatibility**: Following the project's "No Legacy Support" policy, there are no aliases or compatibility layers.

---

## Future Enhancements

### Planned Features

1. **Article Builder**: Complete implementation with proper serialization
2. **Reading Order Detection**: Detect column structure and reading order
3. **Quality Metrics**: Assess detection quality automatically
4. **Batch Visualization**: Process multiple pages in parallel
5. **Interactive Viewer**: Web UI for browsing results

### Potential Extensions (enabled by generalization)

1. **Advertisement Detection**: Match ads to their related articles
2. **Continuation Detection**: Match "continued on page X" markers
3. **Cross-References**: Match footnotes to their reference locations
4. **Image Groups**: Match related images based on proximity
5. **Multi-Column Layout**: Match columns based on spatial relationships

---

## Testing

```bash
# Quick test on small sample
newspaper-explorer analyze layout detect --source der_tag --year 1902 --limit 5

# Verify with visualization
newspaper-explorer analyze layout visualize --source der_tag --year 1902 --limit 5

# Test extraction with filtering
newspaper-explorer analyze layout extract-pictures \
    --source der_tag \
    --year 1902 \
    --limit 5 \
    --exclude-top-percent 15

# Check outputs
ls -lh results/der_tag/layout/
ls -lh results/der_tag/layout/visualizations/
ls -lh results/der_tag/layout/pictures/
```

---

## Dependencies

Required packages (already in `requirements.txt`):
- `ultralytics` - YOLOv11 inference
- `opencv-python` (cv2) - Image processing
- `numpy` - Array operations
- `polars` - DataFrame operations
- `huggingface_hub` - Model downloads
- `pillow` - Image handling
- `tqdm` - Progress tracking

---

## References

- **YOLOv11 Model**: [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout)
- **Dataset**: DocLayNet (11 document structure categories)
- **Original hackathon code**: `__scrap/hackathon/`

---

## Troubleshooting

### Common Issues

**Issue**: Low detection accuracy
- **Solution**: Use larger model size (`--model-size medium`)
- **Solution**: Check image quality and resolution

**Issue**: Too many false positive pictures (headers/logos)
- **Solution**: Use coordinate filtering (`--exclude-top-percent 15`)
- **Solution**: Adjust minimum size constraints

**Issue**: Captions not matching correctly
- **Solution**: Adjust `search_radius` parameter
- **Solution**: Try different `relative_position` values
- **Solution**: Lower `overlap_threshold`

**Issue**: Slow processing
- **Solution**: Use smaller model size (`--model-size nano`)
- **Solution**: Enable GPU acceleration (CUDA)
- **Solution**: Increase batch size

**Issue**: Out of memory
- **Solution**: Reduce batch size
- **Solution**: Use smaller model
- **Solution**: Process fewer pages at once
