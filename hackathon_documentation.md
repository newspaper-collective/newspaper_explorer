# Hackathon Documentation

## Project Submission

**Project code/prototype:**
→ https://github.com/newspaper-collective/newspaper_explorer

**Title:**
→ Newspaper Explorer

**License:**
→ MIT License

**Image:**
→ docs/ui/screenshots/06_issue_reader.png (or choose another from docs/ui/screenshots/ or docs/hackathon/screenshots/)

**Description (250 words max):**

Newspaper Explorer is a toolkit for exploring historical newspapers through computational analysis. Built during the culture.explore(data) hackathon at Staatsbibliothek zu Berlin and refined since, it provides tools to explore large newspaper datasets.

**Data Used:** The "Der Tag" newspaper collection (1900-1920) from the Staatsbibliothek zu Berlin, containing ~148,000 ALTO XML files with 61+ million text lines from 135,000 page images, totaling over 10 GB of compressed XML and 200 GB of JPEG images.

**Core Idea:** Historical newspaper archives contain thousands of pages locked in ALTO XML format. Standard tools connecting raw data to modern NLP, computer vision, and LLM workflows are scarce. Processing data at this scale required specialized approaches: parallel parsing with Polars DataFrames, DuckDB for SQL queries on multi-GB datasets without loading into memory, and resume-capable pipelines that track processed files. Newspaper Explorer bridges this gap with a unified pipeline that downloads, parses, preprocesses, and analyzes newspaper data efficiently.

**Features:** The toolkit combines diverse computational methods—GLiNER for named entity extraction, YOLOv11 for layout detection, emotion classification using a fine-tuned BERT model from Universität Würzburg, LLM-based topic modeling, and keyword extraction. A Vue/FastAPI web interface enables interactive exploration with entity timelines, image galleries, analysis visualizations, and full-text search.

**Impact:** Researchers can take a first step into unknown datasets, generating visualizations and insights that surface patterns and possibilities before specific research questions are formulated. A publication of the complete analysis results for the "Der Tag" corpus on zenodo is planned after some further optimization of the codebase. 