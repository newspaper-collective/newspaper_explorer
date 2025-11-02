# Layout Analysis Generalization

## Overview

The layout analysis modules have been generalized from image-specific operations to work with **any detected element type**. This makes the code more reusable and flexible.

## Changes

### 1. `RegionExtractor` (formerly `ImageExtractor`)

**Purpose**: Extract and crop any type of detected region from page images.

**Key Changes**:
- ✅ Renamed `ImageExtractor` → `RegionExtractor`
- ✅ `extract_images()` → `extract_regions()` with `region_type` parameter
- ✅ `save_image_metadata()` → `save_region_metadata()` (includes `class_name`)
- ✅ Works with any detection type: images, text, headlines, figures, etc.

**Usage**:
```python
from newspaper_explorer.analyze.layout.image_extractor import RegionExtractor

extractor = RegionExtractor(padding=5)

# Extract images
images = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type="image",  # Filter by type
)

# Extract headlines
headlines = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type="headline",
)

# Extract multiple types
visual_elements = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type=["image", "figure", "chart"],  # Can pass list
)

# Extract all
all_regions = extractor.extract_regions(
    detections=all_detections,
    page_layout=page_layout,
    output_dir=output_dir,
    region_type=None,  # No filtering
)
```

### 2. `ProximityMatcher` (formerly `CaptionMatcher`)

**Purpose**: Match any element type to any other element type based on spatial proximity.

**Key Changes**:
- ✅ Renamed `CaptionMatcher` → `ProximityMatcher`
- ✅ `caption_position` → `relative_position` (more expressive)
- ✅ Added support for `left`, `right`, `any` positions (not just `above`/`below`)
- ✅ Generic `match_elements()` method (source → target)
- ✅ `apply_matches()` with configurable `target_attr`

**Usage**:
```python
from newspaper_explorer.analyze.layout.caption_matching import ProximityMatcher

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

## No Backward Compatibility

Following the project's **"No Legacy Support"** policy:
- ❌ No `ImageExtractor` alias
- ❌ No `CaptionMatcher` alias
- ❌ No `extract_images()` wrapper method
- ❌ No `caption_position` parameter support
- ✅ CLI updated to use new API directly
- ✅ Clean break, better code

## Benefits

1. **Reusability**: Extract any region type with the same code
2. **Flexibility**: Match any element to any other element
3. **Spatial Control**: `relative_position` supports 5 modes (above, below, left, right, any)
4. **Type Safety**: `region_type` parameter makes filtering explicit
5. **Future-Proof**: Easy to add new element types without changing code

## Migration Guide

### Old Code (ImageExtractor)
```python
from newspaper_explorer.analyze.layout.image_extractor import ImageExtractor

extractor = ImageExtractor(padding=5)
images = extractor.extract_images(images, page_layout, output_dir)
extractor.save_image_metadata(images, output_path)
```

### New Code (RegionExtractor)
```python
from newspaper_explorer.analyze.layout.image_extractor import RegionExtractor

extractor = RegionExtractor(padding=5)
images = extractor.extract_regions(images, page_layout, output_dir, region_type="image")
extractor.save_region_metadata(images, output_path)
```

### Old Code (CaptionMatcher)
```python
from newspaper_explorer.analyze.layout.caption_matching import CaptionMatcher

matcher = CaptionMatcher(
    search_radius=150,
    caption_position="below",  # Old param name
)
images_with_captions = matcher.match_captions_for_page(page_layout, lines_df)
```

### New Code (ProximityMatcher)
```python
from newspaper_explorer.analyze.layout.caption_matching import ProximityMatcher

matcher = ProximityMatcher(
    search_radius=150,
    relative_position="below",  # New param name
)

# Explicit workflow
matches = matcher.match_elements(images, captions, lines_df, page_id, extract_text=True)
images_with_captions = matcher.apply_matches(matches, target_attr="caption")
```

## Implementation Details

### File Locations
- `src/newspaper_explorer/analysis/layout/image_extractor.py` - Contains `RegionExtractor`
- `src/newspaper_explorer/analysis/layout/caption_matching.py` - Contains `ProximityMatcher`
- `src/newspaper_explorer/cli/analyze/layout/commands.py` - Updated CLI usage

### Key Methods

**RegionExtractor**:
- `extract_regions(detections, page_layout, output_dir, region_type=None)`
- `save_region_metadata(detections, output_path, format="parquet")`
- `_save_region_crop(page_image, detection, output_dir, page_id)`

**ProximityMatcher**:
- `match_elements(source_elements, target_elements, lines_df, page_id, extract_text=True)`
- `apply_matches(matches, target_attr="caption")`
- `_calculate_distance(source_bbox, target_bbox)`
- `_is_valid_position(source_bbox, target_bbox)`

## Future Enhancements

Potential extensions enabled by this generalization:

1. **Advertisement Detection**: Match ads to their related articles
2. **Continuation Detection**: Match "continued on page X" markers
3. **Cross-References**: Match footnotes to their reference locations
4. **Image Groups**: Match related images based on proximity
5. **Multi-Column Layout**: Match columns based on spatial relationships
