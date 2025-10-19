# Image Downloading

The image downloader extracts and downloads high-resolution newspaper page scans from METS XML metadata files.

## Overview

**Purpose**: Download newspaper page images referenced in METS XML files for visual analysis and archival.

**Storage Structure**: `data/raw/{source}/images/{year}/{month}/{day}/`

The directory structure mirrors the XML organization, making it easy to correlate images with their textual content.

## Architecture

### ImageDownloader Class

Located in `data/utils/images.py`, the `ImageDownloader` handles:

1. **METS File Discovery** - Finds all METS XML files (non-fulltext)
2. **URL Extraction** - Parses MAX fileGrp for maximum resolution image URLs
3. **Parallel Downloads** - Multi-threaded downloading with progress tracking
4. **Directory Mirroring** - Maintains year/month/day structure from XML organization
5. **Resume Support** - Automatically skips already downloaded images

### Configuration

Image downloading uses the same source configuration as text loading:

```json
{
  "dataset_name": "der_tag",
  "data_type": "xml_ocr",
  "metadata": {
    "newspaper_title": "Der Tag",
    "format": "ALTO XML"
  }
}
```

## Usage

### CLI Command

```bash
# Download all images for a source
newspaper-explorer data download-images --source der_tag

# Customize parallel workers (default: 8)
newspaper-explorer data download-images --source der_tag --max-workers 16

# Adjust retry attempts (default: 3)
newspaper-explorer data download-images --source der_tag --max-retries 5
```

### Programmatic Usage

```python
from newspaper_explorer.data.utils.images import ImageDownloader

# Initialize downloader
downloader = ImageDownloader(
    source_name="der_tag",
    max_workers=8,
    max_retries=3,
    timeout=30
)

# Download all images
stats = downloader.download_images()

print(f"Downloaded: {stats['downloaded']}")
print(f"Skipped: {stats['skipped']}")
print(f"Failed: {stats['failed']}")
```

### Processing Specific Files

```python
from pathlib import Path

# Process only specific METS files
mets_files = [
    Path("data/raw/der_tag/xml_ocr/1901/01/08/der_tag_1901-01-08.xml"),
    Path("data/raw/der_tag/xml_ocr/1901/01/09/der_tag_1901-01-09.xml"),
]

stats = downloader.download_images(mets_files=mets_files)
```

## Directory Structure

```
data/
└── raw/
    └── der_tag/
        ├── xml_ocr/          # Source XML files
        │   └── 1901/
        │       └── 01/
        │           └── 08/
        │               ├── der_tag_1901-01-08.xml
        │               └── fulltext/
        │                   └── *.xml
        └── images/           # Downloaded images
            └── 1901/
                └── 01/
                    └── 08/
                        ├── FILE_0001_MASTER.jpg
                        ├── FILE_0002_MASTER.jpg
                        └── ...
```

## Image Reference Structure

Images are extracted from METS XML `<fileGrp USE="MAX">` elements:

```xml
<mets:fileGrp USE="MAX">
  <mets:file ID="FILE_0001_MASTER" MIMETYPE="image/jpeg">
    <mets:FLocat xlink:href="https://example.com/image.jpg"/>
  </mets:file>
</mets:fileGrp>
```

- `USE="MAX"`: Maximum resolution images (vs. thumbnails or derivatives)
- `ID`: Used as filename (e.g., `FILE_0001_MASTER.jpg`)
- `xlink:href`: Download URL

## Features

### Optimization

1. **Parallel Downloads**: Multi-threaded downloading (configurable workers)
2. **Resume Capability**: Skips already downloaded images
3. **Atomic Writes**: Uses temporary files to prevent partial downloads
4. **Retry Logic**: Exponential backoff for failed downloads
5. **Progress Tracking**: Visual progress bars with `tqdm`

### Error Handling

- **Timeout**: 30 seconds default (configurable)
- **Retries**: 3 attempts with exponential backoff
- **Logging**: Failed downloads logged with error details
- **Graceful Degradation**: Continues processing after individual failures

### Output Standards

- **CLI**: Uses `click.echo()` for user-facing messages
- **Library**: Uses `logging` module for internal messages
- **Progress**: `tqdm` for download progress bars

## Performance Considerations

### Bandwidth

- Default 8 parallel workers balances speed and server load
- Increase `--max-workers` for faster downloads (if bandwidth allows)
- Decrease if experiencing connection issues

### Disk Space

High-resolution newspaper page images are large:

- ~1-5 MB per page (typical JPEG)
- Full year: ~500-2000 pages = ~0.5-10 GB
- Complete archive (1900-1920): ~10-200 GB

### Processing Time

Depends on:

- Network speed
- Number of parallel workers
- Image resolution and quantity
- Server response time

Typical: ~100-500 images per minute with 8 workers

## Integration with Analysis

Downloaded images can be used for:

1. **Layout Analysis**: Extract regions, columns, graphics
2. **OCR Verification**: Compare OCR text with visual content
3. **Visual Features**: Analyze fonts, formatting, illustrations
4. **Archival**: Complete digital preservation with images + text

See `analysis/layout/` for image-based analysis modules.

## Troubleshooting

### No images found

- Verify METS files exist in `data/raw/{source}/xml_ocr/`
- Check if METS files contain `<fileGrp USE="MAX">`
- Ensure you're not in the `fulltext/` directory

### Downloads failing

- Check network connectivity
- Verify image URLs are accessible
- Increase `--max-retries` for unstable connections
- Reduce `--max-workers` if server is rate-limiting

### Disk space issues

- Check available space before downloading
- Download specific date ranges if needed (custom script)
- Use compression or external storage for large datasets

## Example Workflow

```bash
# 1. Download and extract XML data
newspaper-explorer data download --source der_tag --all
newspaper-explorer data load --source der_tag

# 2. Download images
newspaper-explorer data download-images --source der_tag

# 3. Analyze with images and text together
# (custom analysis combining both data types)
```

## Future Enhancements

Potential improvements:

- [ ] Date range filtering (`--start-date`, `--end-date`)
- [ ] Download only specific resolutions (MAX, DEFAULT, THUMBS)
- [ ] Image preprocessing (resize, format conversion)
- [ ] Incremental updates (check for new issues)
- [ ] Mirror mode (download from alternative CDNs)
- [ ] Checksum verification for downloaded images

## Related Documentation

- [Data Pipeline](DATA.md) - Overall data processing workflow
- [Configuration](CONFIGURATION_PHILOSOPHY.md) - Source configuration
- [CLI Commands](CLI.md) - Complete command reference
