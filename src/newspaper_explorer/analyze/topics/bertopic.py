#!/usr/bin/env python3
"""
Topic modeling script for newspaper pages using BERTopic.
Processes text files from dump/ directory and generates topic analysis CSV.
Adapted to work with individual page files rather than aggregated issues.
"""

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ======================== CONFIG ========================
DUMP_ROOT = Path("/mnt/data/userfiles/westphal/hackathon/dump")
OUTPUT_DIR = Path("/mnt/data/userfiles/westphal/hackathon/output")
CSV_OUT = OUTPUT_DIR / "topics_analysis.csv"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# BERTopic parameters
MIN_TOPIC_SIZE = 3  # Minimum number of documents per topic
N_TOP_TOPICS = 5  # Number of top topics to extract per document group
MIN_TEXT_LENGTH = 100  # Minimum text length to consider


# ======================== HELPERS ========================
def parse_filename(filename: str):
    """
    Parse metadata from filename like: 3074409X_1900-01-02_000_2_H_1_001.txt
    Returns: (date_str, issue_num, part_num, page_num)
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})_\d+_(\d+)_H_(\d+)_(\d+)", filename)
    if match:
        date_str, issue, part, page = match.groups()
        return date_str, issue, part, page
    return None, None, None, None


def group_files_by_issue(dump_dir: Path):
    """
    Group text files by date and issue number.
    Returns dict: {(date, issue): [file_paths]}
    """
    groups = defaultdict(list)

    for txt_file in sorted(dump_dir.glob("*.txt")):
        date_str, issue, part, page = parse_filename(txt_file.name)
        if date_str and issue:
            groups[(date_str, issue)].append(txt_file)

    return groups


def read_and_clean_text(file_path: Path):
    """Read text file and extract clean content, skipping metadata."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        # Skip the header (Source, Date, Total blocks, separator)
        lines = text.split("\n")
        content_lines = []
        in_content = False

        for line in lines:
            # Skip metadata and block headers
            if (
                line.startswith("Source:")
                or line.startswith("Date:")
                or line.startswith("Total blocks:")
                or line.startswith("===")
                or line.startswith("--- Block")
                or line.startswith("Position:")
            ):
                continue

            # Clean and add non-empty lines
            clean_line = line.strip()
            if clean_line:
                content_lines.append(clean_line)

        return " ".join(content_lines)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def aggregate_issue_text(file_paths):
    """Combine all pages of an issue into one document."""
    texts = []
    for fp in sorted(file_paths):
        text = read_and_clean_text(fp)
        if len(text) >= MIN_TEXT_LENGTH:
            texts.append(text)

    return " ".join(texts) if texts else ""


def extract_topic_keywords(topic_model, topic_id, top_k=5):
    """Extract top keywords for a topic."""
    words_scores = topic_model.get_topic(topic_id)
    if not words_scores:
        return []
    return [word for word, score in words_scores[:top_k]]


def fallback_keywords(text, top_k=5):
    """Simple fallback when no topics can be formed using term frequency."""
    if not text or len(text) < MIN_TEXT_LENGTH:
        return []

    cv = CountVectorizer(stop_words=None, ngram_range=(1, 2), min_df=1, max_df=0.95)
    try:
        X = cv.fit_transform([text])
        vocab = cv.get_feature_names_out()
        counts = X.toarray().sum(axis=0)
        pairs = sorted(zip(vocab, counts), key=lambda t: t[1], reverse=True)
        return [w for w, _ in pairs[:top_k]]
    except Exception:
        return []


# ======================== MAIN PROCESSING ========================
def main():
    print("Starting topic modeling on newspaper dump files...")
    print(f"Input directory: {DUMP_ROOT}")
    print(f"Output CSV: {CSV_OUT}")

    # Check if dump directory exists
    if not DUMP_ROOT.exists():
        raise FileNotFoundError(f"Dump directory not found: {DUMP_ROOT}")

    # Group files by issue
    print("\nGrouping files by issue...")
    issue_groups = group_files_by_issue(DUMP_ROOT)
    print(f"Found {len(issue_groups)} issues")

    if not issue_groups:
        raise FileNotFoundError("No valid text files found in dump directory")

    # Initialize models
    print("\nInitializing BERTopic models...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vectorizer_model = CountVectorizer(stop_words=None, ngram_range=(1, 2), min_df=2)

    # Process each issue
    rows = []

    for (date_str, issue_num), file_paths in tqdm(
        issue_groups.items(), desc="Processing issues"
    ):
        # Parse date
        try:
            year, month, day = date_str.split("-")
        except:
            year, month, day = "", "", ""

        # Aggregate all pages into one document
        full_text = aggregate_issue_text(file_paths)

        if len(full_text) < MIN_TEXT_LENGTH:
            # Fallback for very short documents
            keywords = fallback_keywords(full_text, top_k=5)
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "issue": issue_num,
                    "num_pages": len(file_paths),
                    "topics": "|".join(keywords[:N_TOP_TOPICS]),
                    "subtopics": "|".join(keywords[:5]),
                    "method": "fallback",
                }
            )
            continue

        # Split into smaller chunks for better topic detection
        # Using simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", full_text)

        # Create chunks of ~5 sentences each
        chunks = []
        chunk_size = 5
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i : i + chunk_size])
            if len(chunk) >= MIN_TEXT_LENGTH:
                chunks.append(chunk)

        if len(chunks) < MIN_TOPIC_SIZE:
            # Not enough chunks for topic modeling
            keywords = fallback_keywords(full_text, top_k=5)
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "issue": issue_num,
                    "num_pages": len(file_paths),
                    "topics": "|".join(keywords[:N_TOP_TOPICS]),
                    "subtopics": "|".join(keywords[:5]),
                    "method": "fallback",
                }
            )
            continue

        # Perform topic modeling
        try:
            topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer_model,
                min_topic_size=MIN_TOPIC_SIZE,
                nr_topics=None,
                calculate_probabilities=False,
                verbose=False,
            )

            topics, _ = topic_model.fit_transform(chunks)

            # Get topic information
            info = topic_model.get_topic_info()
            info = info[info["Topic"] != -1].sort_values("Count", ascending=False)

            if info.empty:
                # No topics found
                keywords = fallback_keywords(full_text, top_k=5)
                topic_labels = keywords[:N_TOP_TOPICS]
                subtopic_keywords = keywords[:5]
                method = "bertopic_notopics"
            else:
                # Extract top topics
                top_topic_ids = info["Topic"].head(N_TOP_TOPICS).tolist()
                topic_labels = []

                for tid in top_topic_ids:
                    keywords = extract_topic_keywords(topic_model, tid, top_k=2)
                    topic_labels.append(" ".join(keywords))

                # Get subtopics from the most dominant topic
                dominant_tid = top_topic_ids[0]
                subtopic_keywords = extract_topic_keywords(
                    topic_model, dominant_tid, top_k=5
                )
                method = "bertopic"

            rows.append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "issue": issue_num,
                    "num_pages": len(file_paths),
                    "topics": "|".join(topic_labels),
                    "subtopics": "|".join(subtopic_keywords),
                    "method": method,
                }
            )

        except Exception as e:
            print(f"\nError processing {date_str} issue {issue_num}: {e}")
            # Fallback
            keywords = fallback_keywords(full_text, top_k=5)
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "issue": issue_num,
                    "num_pages": len(file_paths),
                    "topics": "|".join(keywords[:N_TOP_TOPICS]),
                    "subtopics": "|".join(keywords[:5]),
                    "method": "error_fallback",
                }
            )

    # Create and save DataFrame
    print("\nCreating output CSV...")
    df = pd.DataFrame(
        rows,
        columns=[
            "year",
            "month",
            "day",
            "issue",
            "num_pages",
            "topics",
            "subtopics",
            "method",
        ],
    )
    df.sort_values(["year", "month", "day", "issue"], inplace=True)
    df.to_csv(CSV_OUT, index=False, encoding="utf-8")

    # Print summary
    print(f"\n✅ Successfully processed {len(rows)} issues")
    print(f"   Output saved to: {CSV_OUT}")
    print(f"\nMethod breakdown:")
    print(df["method"].value_counts())
    print(
        f"\nDate range: {df['year'].min()}-{df['month'].min()}-{df['day'].min()} to "
        + f"{df['year'].max()}-{df['month'].max()}-{df['day'].max()}"
    )


if __name__ == "__main__":
    main()
