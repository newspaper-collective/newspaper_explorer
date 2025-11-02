# 📊 Data Loader Implementation Summary

## ✅ What We Built

A high-performance ALTO XML parser for newspaper data with:

### Core Features

- ✅ **Line-level granularity** - Maximum flexibility for analysis
- ✅ **Parallel processing** - Uses all CPU cores (multiprocessing)
- ✅ **Polars DataFrames** - 2-5x faster than Pandas
- ✅ **Parquet format** - Compressed, columnar storage
- ✅ **Automatic namespace detection** - Handles different ALTO versions
- ✅ **Progress tracking** - Real-time tqdm progress bars
- ✅ **Comprehensive metadata** - Dates, coordinates, IDs

### Components Created

1. **`src/newspaper_explorer/data/loading.py`**

   - `TextLine` dataclass - Single text line representation
   - `ALTOParser` - Fast XML parser with namespace detection
   - `DataLoader` - High-level API with parallel processing

2. **CLI command: `newspaper-explorer data load`**

   - Load directories of XML files
   - Configurable workers and file patterns
   - Outputs Parquet files

3. **Documentation**

   - `docs/LOADING.md` - Comprehensive guide
   - Usage examples and performance tips

4. **Example script**

   - `examples/load_alto_data.py` - Working example

5. **Tests**
   - `tests/data/test_loading.py` - Unit tests

## 📋 DataFrame Schema

Each row = one text line from ALTO XML:

```python
{
    "line_id": "3074409X_1901-01-02_..._page1_block1_line1",
    "text": "Cleaned OCR text",
    "text_block_id": "block_1",
    "page_id": "page_1",
    "filename": "3074409X_1901-01-02_000_2_H_1_009.xml",
    "date": datetime(1901, 1, 2),
    "year": 1901,
    "month": 1,
    "day": 2,
    "x": 100,           # Line position
    "y": 200,
    "width": 500,
    "height": 20
}
```

## 🚀 Usage

### CLI

```bash
# Load all XML files
newspaper-explorer data load data/raw/der_tag/xml_ocr data/extracted/lines.parquet

# Test with 100 files
newspaper-explorer data load data/raw/der_tag/xml_ocr test.parquet --max-files 100 --workers 4
```

### Python API

```python
from newspaper_explorer.data.loading import DataLoader

loader = DataLoader(max_workers=4)
df = loader.load_directory(
    directory=Path("data/raw/der_tag/xml_ocr"),
    output_parquet=Path("lines.parquet")
)

# Load saved Parquet
df = DataLoader.load_parquet("lines.parquet")

# Filter by date
df_1901 = df.filter(df["year"] == 1901)

# Aggregate to text blocks
blocks = df.group_by(["filename", "text_block_id"]).agg([
    df["text"].str.concat(" ").alias("full_text"),
    df["line_id"].count().alias("num_lines"),
])
```

## 🎯 Why Line-Level?

**Flexibility is key!**

1. ✅ Can **aggregate UP** → lines → blocks → pages → issues
2. ❌ Cannot **disaggregate DOWN** if stored as blocks
3. ✅ Enables **layout analysis** (columns, spacing, positioning)
4. ✅ Preserves **spatial structure** from ALTO
5. ✅ Allows **OCR quality analysis** per line

## ⚡ Performance Optimizations

### 1. Parallel XML Parsing

- Uses `ProcessPoolExecutor` for CPU-bound work
- Default: `CPU count - 1` workers
- Each worker parses files independently

### 2. Polars Instead of Pandas

| Operation          | Pandas | Polars | Speedup         |
| ------------------ | ------ | ------ | --------------- |
| DataFrame creation | Slow   | Fast   | **2-3x**        |
| Filtering          | Slow   | Fast   | **5-10x**       |
| Parquet I/O        | Slow   | Fast   | **3-5x**        |
| Memory usage       | High   | Low    | **30-50% less** |

### 3. Parquet Format

- Columnar storage (read only needed columns)
- Built-in compression (zstd)
- 3-5x faster than CSV
- 50-80% smaller files

### 4. Batch Processing

- Processes files in chunks
- Saves incrementally to Parquet
- Lower memory footprint

## 📦 Dependencies Added

```toml
dependencies = [
    "polars>=0.20.0",    # Fast DataFrame library
    "lxml>=5.0.0",       # XML parsing
    "pyarrow>=15.0.0",   # Parquet support
]
```

## 🔄 Typical Workflow

```
1. Download & extract data
   └─→ newspaper-explorer data download --all

2. Load ALTO XML → Parquet
   └─→ newspaper-explorer data load data/raw/der_tag/xml_ocr lines.parquet

3. Analysis in Python/Notebook
   └─→ df = DataLoader.load_parquet("lines.parquet")
   └─→ Filter, aggregate, analyze...
```

## 📈 Next Steps

With data loaded, you can now:

1. **Text Analysis**

   - Topic modeling (LDA, BERTopic)
   - Named entity recognition (spaCy, Flair)
   - Sentiment analysis
   - N-gram analysis

2. **Layout Analysis**

   - Column detection
   - Article segmentation
   - Advertisement detection
   - Page structure analysis

3. **Temporal Analysis**

   - Topic trends over time
   - Seasonal patterns
   - Event detection

4. **Quality Analysis**
   - OCR confidence scores
   - Missing dates/issues
   - Data completeness

## 💡 Tips

### Fast Filtering

```python
# Lazy loading - only reads needed columns
df = pl.scan_parquet("lines.parquet") \
    .select(["text", "date"]) \
    .filter(pl.col("year") == 1901) \
    .collect()
```

### Memory-Efficient Iteration

```python
# Process in chunks
for chunk in df.iter_slices(n_rows=10000):
    process(chunk)
```

### Resume Interrupted Processing

```python
# Check which files are already processed
processed_files = df["filename"].unique().to_list()
remaining_files = [f for f in all_files if f.name not in processed_files]

# Process only remaining
loader.load_files(remaining_files, output_parquet="more_lines.parquet")
```

## 🔍 File Structure

```
src/newspaper_explorer/data/
├── loading.py          # Main data loader (NEW!)
├── download.py         # Zenodo downloader
├── cleaning.py         # Data cleaning
└── fixes.py           # Error corrections

examples/
└── load_alto_data.py   # Usage example (NEW!)

docs/
├── LOADING.md         # Comprehensive guide (NEW!)
├── DATA.md
└── CLI.md

tests/data/
├── test_loading.py     # Unit tests (NEW!)
└── test_download.py
```

## ✨ Key Design Decisions

1. **Line-level over block-level**

   - Pro: Maximum flexibility, can aggregate later
   - Con: Larger dataset (acceptable trade-off)

2. **Polars over Pandas**

   - Pro: 2-5x faster, better memory
   - Con: Different API (but similar enough)

3. **Parquet over CSV**

   - Pro: Much faster, smaller, typed
   - Con: Binary format (not human-readable)

4. **Multiprocessing over threading**
   - Pro: True parallelism for CPU-bound XML parsing
   - Con: More overhead (acceptable for large files)

## 🎉 Ready to Use!

The data loader is now fully functional and optimized for large-scale newspaper processing. Install dependencies and start loading:

```bash
# Install dependencies
pip install -e .

# Load your data
newspaper-explorer data load data/raw/der_tag/xml_ocr data/extracted/lines.parquet

# Start analyzing!
python -c "from newspaper_explorer.data.loading import DataLoader; df = DataLoader.load_parquet('data/extracted/lines.parquet'); print(df.describe())"
```
