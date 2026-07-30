#!/usr/bin/env bash
set -e

echo "=== Step 1/4: Preprocessing ==="

echo "--- preprocess-all (textblocks: keywords, topics, entities) ---"
newspaper-explorer data preprocess-all --source der_tag \
    --input-types textblocks --presets keywords,topics,entities

echo "--- preprocess-all (lines: keywords) ---"
newspaper-explorer data preprocess-all --source der_tag \
    --input-types lines --presets keywords

echo "=== Step 2/4: Keywords ==="

echo "--- yake ---"
newspaper-explorer analyze keywords yake --source der_tag \
    --input-file data/preprocessed/der_tag/textblocks/keywords_textblocks/textblocks.parquet

echo "--- rake ---"
newspaper-explorer analyze keywords rake --source der_tag \
    --input-file data/preprocessed/der_tag/textblocks/keywords_textblocks/textblocks.parquet

echo "--- keybert ---"
newspaper-explorer analyze keywords keybert --source der_tag \
    --input-file data/preprocessed/der_tag/textblocks/keywords_textblocks/textblocks.parquet

echo "--- tfidf (lines) ---"
newspaper-explorer analyze keywords tfidf --source der_tag \
    --input-file data/preprocessed/der_tag/lines/keywords_lines/lines.parquet

echo "=== Step 3/4: Topics ==="

echo "--- lda train ---"
newspaper-explorer analyze topics lda --source der_tag \
    --input-file data/preprocessed/der_tag/textblocks/topics_textblocks/textblocks.parquet \
    --mode train --num-topics 15 --passes 15 --force-retrain

echo "--- lda topics ---"
newspaper-explorer analyze topics lda --source der_tag \
    --input-file data/preprocessed/der_tag/textblocks/topics_textblocks/textblocks.parquet \
    --mode topics --num-topics 15 --top-k 15

echo "--- lda documents ---"
newspaper-explorer analyze topics lda --source der_tag \
    --input-file data/preprocessed/der_tag/textblocks/topics_textblocks/textblocks.parquet \
    --mode documents --num-topics 15

echo "=== Step 4/4: Entities ==="

echo "--- entity extraction ---"
newspaper-explorer analyze entities extract --source der_tag \
    --input-file data/preprocessed/der_tag/textblocks/entities_textblocks/textblocks.parquet

echo "=== All done ==="
