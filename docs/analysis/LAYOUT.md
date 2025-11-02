# Layout Analysis Integration

This document describes the integration of YOLOv11 layout analysis from the hackathon code into the main codebase.

## Overview

The layout analysis system provides:
1. **Detection** - YOLOv11-based detection of 11 document element types
2. **Text Matching** - Unified OCR text extraction for any detected element
3. **Image Extraction** - Extract images with matched captions
4. **Headline Matching** - Match headlines to OCR text
5. **Article Reconstruction** - Build articles from headlines + text blocks
6. **Visualization** - Debug visualizations of detected regions

## Architecture

```
analysis/layout/
├── detector.py         # YOLOv11 wrapper (library-style, no CLI)
├── text_linker.py      # Universal text extraction (DataFrame-based, FAST)
├── headline_matcher.py # Headline-specific wrapper using TextLinker
├── image_extractor.py  # Image + caption extraction using TextLinker
├── article_builder.py  # Article reconstruction
├── visualizer.py       # Debug visualizations
└── schemas.py          # Data schemas

cli/layout.py           # Layout analysis CLI commands
```

## Key Design Decisions

### 1. Unified Text Linking (DataFrame-Based)

**Problem**: Extracting OCR text for captions and headlines uses the same technique (bounding box overlap).

**Solution**: Created `TextLinker` class that works with pre-parsed DataFrames for FAST batch processing:

```python
from newspaper_explorer.analyze.layout.text_linker import TextLinker
from newspaper_explorer.data.loading.loader import DataLoader

# Load parquet ONCE (contains ALL pages)
df = DataLoader.load_parquet("data/processed/der_tag/text/der_tag_lines.parquet")

linker = TextLinker(overlap_threshold=0.3)

# Process many pages - NO XML parsing overhead!
for page_id in page_ids:
    matched_headlines = linker.link_detections_to_text(
        detections=headlines,
        lines_df=df,  # Reuse same DataFrame
        page_id=page_id
    )
    
    matched_captions = linker.link_detections_to_text(
        detections=captions,
        lines_df=df,
        page_id=page_id
    )

```

### 2. Visualization for Debugging

**Problem**: Need to debug and verify detected regions.

**Solution**: Created `LayoutVisualizer` class with multiple visualization modes:

```python
from newspaper_explorer.analyze.layout.visualizer import LayoutVisualizer

visualizer = LayoutVisualizer(show_text=True, show_confidence=True)

# Single view with all elements
visualizer.visualize_page(page_layout, "output.jpg")

# Comparison view (side-by-side panels for each element type)
visualizer.visualize_comparison(page_layout, "comparison.jpg")

# Quick visualization
from newspaper_explorer.analyzelayout.visualizer import quick_visualize
quick_visualize(page_layout, "debug.jpg", element_types=["Title", "Picture"])
```

Features:
- Color-coded bounding boxes by element type
- Confidence scores
- Matched OCR text overlay
- Legend with element counts
- Comparison mode (separate panels for headlines/images/captions/tables)

### 3. Library-First Design

All modules follow the library-first pattern:
- No `print()` statements (use `logging`)
- No CLI logic in library code
- Configuration-driven
- Polars DataFrames for data
- Type hints throughout

## CLI Commands

All commands follow the pattern: `newspaper-explorer layout <command> --source <name>`

### 1. Detect Layout Elements

```bash
# Detect all layout elements in a year
newspaper-explorer layout detect --source der_tag --year 1902

# Use different model size (nano/small/medium)
newspaper-explorer layout detect --source der_tag --model-size nano

# Limit processing for testing
newspaper-explorer layout detect --source der_tag --year 1902 --limit 10
```

**Output**: JSON files in `results/der_tag/layout/` with detected elements

### 2. Visualize Detections (NEW)

```bash
# Visualize specific page
newspaper-explorer layout visualize --source der_tag --page-id 1902_01_01_001

# Visualize first 10 pages of a year
newspaper-explorer layout visualize --source der_tag --year 1902 --limit 10

# Show only headlines and images
newspaper-explorer layout visualize --source der_tag --page-id 1902_01_01_001 \
    --element-types title --element-types picture

# Create comparison view
newspaper-explorer layout visualize --source der_tag --page-id 1902_01_01_001 --comparison

# Without OCR text overlay
newspaper-explorer layout visualize --source der_tag --page-id 1902_01_01_001 --no-show-text
```

**Output**: Annotated images in `results/der_tag/layout/visualizations/`

### 3. Extract Images with Captions

```bash
# Extract all images with caption matching
newspaper-explorer layout extract-images --source der_tag --year 1902

# Caption position options
newspaper-explorer layout extract-images --source der_tag --year 1902 \
    --caption-position below  # or: above, both

# Skip saving image crops (metadata only)
newspaper-explorer layout extract-images --source der_tag --year 1902 --no-save-crops
```

**Output**: 
- Cropped images: `results/der_tag/layout/images/`
- Metadata: `results/der_tag/layout/der_tag_images_metadata.parquet`

### 4. Match Headlines to OCR

```bash
# Match detected headlines to OCR text from ALTO XML
newspaper-explorer layout match-headlines --source der_tag --year 1902

# Adjust overlap threshold
newspaper-explorer layout match-headlines --source der_tag --year 1902 \
    --overlap-threshold 0.5
```

**Output**: `results/der_tag/layout/der_tag_headlines.parquet`

### 5. Build Articles (TODO)

```bash
# Reconstruct articles from headlines and text blocks
newspaper-explorer layout build-articles --source der_tag --year 1902
```

This command needs full implementation (headline object reconstruction from DataFrame).

## Detected Element Types

YOLOv11 model detects 11 DocLayNet categories:

| Element | Description | Use Case |
|---------|-------------|----------|
| **Title** | Main article headlines | Article reconstruction |
| **Section-header** | Sub-headlines | Article structure |
| **Picture** | Images/photos | Visual content extraction |
| **Caption** | Image captions | Image metadata |
| **Table** | Tables | Structured data extraction |
| **Text** | Body text paragraphs | Article content |
| **List-item** | List elements | Structured content |
| **Formula** | Mathematical formulas | Scientific content |
| **Page-header** | Page headers | Layout understanding |
| **Page-footer** | Page footers | Layout understanding |
| **Footnote** | Footnotes | Supplementary content |

## Workflow Example

Complete workflow for processing a year:

```bash
# 1. Detect layout elements
newspaper-explorer layout detect --source der_tag --year 1902 --model-size medium

# 2. Visualize sample pages to verify detection quality
newspaper-explorer layout visualize --source der_tag --year 1902 --limit 5 --comparison

# 3. Extract images with captions
newspaper-explorer layout extract-images --source der_tag --year 1902

# 4. Match headlines to OCR text
newspaper-explorer layout match-headlines --source der_tag --year 1902

# 5. Build articles (when implemented)
newspaper-explorer layout build-articles --source der_tag --year 1902
```

## Data Schemas

### Detection
```python
@dataclass
class Detection:
    detection_id: str
    class_name: str          # Element type
    confidence: float        # Detection confidence (0-1)
    bbox: BoundingBox       # Bounding box coordinates
    page_id: str
    
    # Matched from ALTO XML
    text_content: Optional[str] = None
    alto_elements: List[str] = field(default_factory=list)
    
    # For images
    image_path: Optional[str] = None
    caption: Optional["Detection"] = None
    caption_text: Optional[str] = None
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

### Article
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

## Integration with Existing Data Pipeline

The layout analysis integrates seamlessly with the existing pipeline:

```
Download → Parse ALTO/METS → Polars DataFrame
                                    ↓
                            Layout Detection
                                    ↓
                          Text Matching (ALTO)
                                    ↓
                       Headlines/Images/Articles
```

**Key Integration Points**:
1. Uses existing Polars DataFrames from `DataLoader`
2. Reads ALTO XML coordinates directly
3. Outputs to standard results directory
4. Follows same configuration patterns

## Performance Considerations

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

### Memory

- Layout detection: ~2-4GB GPU memory
- Text matching: Minimal overhead
- Visualization: Load full page images (can be large)

## Future Enhancements

1. **Article Builder**: Complete implementation with proper serialization
2. **Caption-Image Association**: Improve spatial reasoning
3. **Reading Order**: Detect column structure and reading order
4. **Quality Metrics**: Assess detection quality automatically
5. **Batch Visualization**: Process multiple pages in parallel
6. **Interactive Viewer**: Web UI for browsing results

## Migration from Hackathon Code

The integration refactored the hackathon code with these improvements:

| Hackathon Code | New Integration | Improvement |
|----------------|-----------------|-------------|
| `yolov11_layout.py` (CLI script) | `detector.py` (library) | No CLI mixing, proper logging |
| Separate headline/caption matching | `text_matcher.py` (unified) | DRY principle, reusable |
| No visualization | `visualizer.py` | Debug & verify detections |
| Manual JSON loading | Polars DataFrame integration | Consistent with pipeline |
| Hardcoded paths | Configuration-driven | Follows project patterns |

## Testing

```bash
# Test detection on small sample
newspaper-explorer layout detect --source der_tag --year 1902 --limit 5

# Verify with visualization
newspaper-explorer layout visualize --source der_tag --year 1902 --limit 5

# Check outputs
ls -lh results/der_tag/layout/
ls -lh results/der_tag/layout/visualizations/
```

## Dependencies

Required packages (already in requirements.txt):
- `ultralytics` - YOLOv11 inference
- `opencv-python` (cv2) - Image processing
- `numpy` - Array operations
- `polars` - DataFrame operations
- `huggingface_hub` - Model downloads

## References

- YOLOv11 Model: [Armaggheddon/yolo11-document-layout](https://huggingface.co/Armaggheddon/yolo11-document-layout)
- Dataset: DocLayNet (11 document structure categories)
- Original hackathon code: `src/hackathon/`
