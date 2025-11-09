"""
Article Builder - Reconstruct newspaper articles from headlines and text blocks.

This module uses detected headlines as anchors to group following text blocks
into coherent articles.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
import polars as pl

from newspaper_explorer.analyze.layout.schemas import (
    Article,
    Headline,
    Detection,
    BoundingBox,
    PageLayout,
)

logger = logging.getLogger(__name__)


class ArticleBuilder:
    """
    Reconstructs newspaper articles from headlines and text blocks.

    Strategy:
    1. Use headlines as article anchors
    2. Find text blocks below/near each headline
    3. Group text blocks by proximity and reading order
    4. Associate images/tables/captions within article bounds
    """

    def __init__(
        self,
        vertical_threshold: int = 100,
        horizontal_threshold: int = 50,
        min_text_length: int = 20,
    ):
        """
        Initialize the ArticleBuilder.

        Args:
            vertical_threshold: Max vertical distance to consider text as part of article (pixels)
            horizontal_threshold: Max horizontal distance for column detection (pixels)
            min_text_length: Minimum text length to consider a valid article (characters)
        """
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        self.min_text_length = min_text_length

        logger.info(
            f"ArticleBuilder initialized: vertical_threshold={vertical_threshold}, "
            f"horizontal_threshold={horizontal_threshold}"
        )

    def build_articles(
        self,
        headlines: List[Headline],
        page_layout: PageLayout,
        lines_df: pl.DataFrame,
    ) -> List[Article]:
        """
        Build articles from headlines and text blocks.

        Args:
            headlines: List of matched headlines
            page_layout: PageLayout with all detected elements
            lines_df: Polars DataFrame with ALTO text lines

        Returns:
            List of reconstructed Article objects
        """
        if not headlines:
            logger.debug("No headlines provided, cannot build articles")
            return []

        logger.info(f"Building articles from {len(headlines)} headlines in {page_layout.page_id}")

        # Sort headlines by vertical position (top to bottom)
        sorted_headlines = sorted(headlines, key=lambda h: h.detection.bbox.y1)

        articles = []

        for idx, headline in enumerate(sorted_headlines):
            # Determine article bounds (between this headline and next)
            y_start = headline.detection.bbox.y2  # Below headline

            if idx < len(sorted_headlines) - 1:
                # Stop before next headline
                y_end = sorted_headlines[idx + 1].detection.bbox.y1
            else:
                # Last article goes to bottom of page
                y_end = float("inf")

            # Find text blocks in this region
            article_text_blocks = self._find_text_blocks_in_region(
                headline.detection.bbox,
                y_start,
                y_end,
                lines_df,
                page_layout.page_id,
            )

            if not article_text_blocks:
                logger.debug(f"No text blocks found for headline: {headline.ocr_text[:30]}...")
                continue

            # Combine text
            full_text = " ".join(article_text_blocks["text"])

            if len(full_text) < self.min_text_length:
                logger.debug(f"Article too short ({len(full_text)} chars), skipping")
                continue

            # Find associated media
            article_bbox = self._calculate_article_bbox(
                headline.detection.bbox,
                article_text_blocks,
            )

            images = self._find_elements_in_region(
                page_layout.images,
                article_bbox,
                y_start,
                y_end,
            )

            tables = self._find_elements_in_region(
                page_layout.tables,
                article_bbox,
                y_start,
                y_end,
            )

            # Create Article
            article = Article(
                article_id=f"{page_layout.page_id}_article_{idx}",
                headline=headline,
                text_blocks=article_text_blocks["text_block_ids"],
                full_text=full_text,
                page_id=page_layout.page_id,
                year=page_layout.year,
                date=page_layout.date,
                newspaper_title=page_layout.newspaper_title,
                images=images,
                tables=tables,
                bbox=article_bbox,
            )

            articles.append(article)
            logger.debug(
                f"Built article: headline='{headline.ocr_text[:50]}...', "
                f"text_length={len(full_text)}, images={len(images)}, tables={len(tables)}"
            )

        logger.info(f"Built {len(articles)} articles from {len(headlines)} headlines")
        return articles

    def _find_text_blocks_in_region(
        self,
        headline_bbox: BoundingBox,
        y_start: float,
        y_end: float,
        lines_df: pl.DataFrame,
        page_id: str,
    ) -> Dict[str, List]:
        """
        Find text blocks in a vertical region below a headline.

        Args:
            headline_bbox: Bounding box of the headline
            y_start: Start Y coordinate
            y_end: End Y coordinate
            lines_df: DataFrame with text lines
            page_id: Page identifier

        Returns:
            Dictionary with 'text' and 'text_block_ids' lists
        """
        # Filter lines for this page and region
        page_df = lines_df.filter(
            (pl.col("filename").str.contains(page_id))
            & (pl.col("y") >= y_start)
            & (pl.col("y") < y_end)
        )

        if page_df.is_empty():
            return {"text": [], "text_block_ids": []}

        # Check horizontal proximity (same column as headline)
        x_center = headline_bbox.center_x
        page_df = page_df.filter(
            (pl.col("x") >= x_center - self.horizontal_threshold)
            & (pl.col("x") <= x_center + self.horizontal_threshold)
        )

        if page_df.is_empty():
            return {"text": [], "text_block_ids": []}

        # Sort by vertical position
        page_df = page_df.sort("y")

        # Group by text_block_id and aggregate
        if "text_block_id" in page_df.columns:
            text_blocks = (
                page_df.group_by("text_block_id")
                .agg([pl.col("text").str.concat(" "), pl.col("y").min()])
                .sort("y")
            )

            text_list = text_blocks["text"].to_list()
            text_block_ids = text_blocks["text_block_id"].to_list()
        else:
            # Fallback if no text_block_id
            text_list = page_df["text"].to_list()
            text_block_ids = []

        return {"text": text_list, "text_block_ids": text_block_ids}

    def _calculate_article_bbox(
        self,
        headline_bbox: BoundingBox,
        text_blocks: Dict[str, List],
    ) -> BoundingBox:
        """
        Calculate bounding box encompassing entire article.

        Args:
            headline_bbox: Headline bounding box
            text_blocks: Dictionary with text block data

        Returns:
            BoundingBox covering the article
        """
        # Start with headline bbox
        x1 = headline_bbox.x1
        y1 = headline_bbox.y1
        x2 = headline_bbox.x2
        y2 = headline_bbox.y2

        # Expand to cover text blocks (would need line coordinates)
        # For now, use vertical extent based on threshold
        y2 = y1 + self.vertical_threshold * len(text_blocks["text"])

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _find_elements_in_region(
        self,
        elements: List[Detection],
        article_bbox: BoundingBox,
        y_start: float,
        y_end: float,
    ) -> List[Detection]:
        """
        Find detected elements (images, tables) within article region.

        Args:
            elements: List of detections
            article_bbox: Article bounding box
            y_start: Start Y coordinate
            y_end: End Y coordinate

        Returns:
            List of detections within region
        """
        in_region = []

        for elem in elements:
            # Check vertical overlap
            if elem.bbox.y1 < y_end and elem.bbox.y2 > y_start:
                # Check horizontal overlap
                if elem.bbox.x1 < article_bbox.x2 and elem.bbox.x2 > article_bbox.x1:
                    in_region.append(elem)

        return in_region

    def save_articles(
        self,
        articles: List[Article],
        output_dir: Path,
        source_name: str,
        format: str = "parquet",
        save_metadata: bool = True,
        metadata_params: Optional[Dict] = None,
    ):
        """
        Save articles to file.

        Args:
            articles: List of articles to save
            output_dir: Output directory
            source_name: Source identifier (e.g., 'der_tag')
            format: Output format ('parquet' or 'json')
            save_metadata: Whether to save metadata.json file
            metadata_params: Additional parameters for metadata
        """
        if not articles:
            logger.warning("No articles to save")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to dictionaries
        articles_data = [article.model_dump(mode="json", exclude_none=True) for article in articles]

        if format == "parquet":
            # Convert to Polars DataFrame
            df = pl.DataFrame(articles_data)
            output_path = output_dir / f"{source_name}_articles.parquet"
            df.write_parquet(output_path)
            logger.info(f"Saved {len(articles)} articles to {output_path}")

        elif format == "json":
            output_path = output_dir / f"{source_name}_articles.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(articles_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(articles)} articles to {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save metadata.json
        if save_metadata:
            metadata_file = output_dir / f"{source_name}_articles_metadata.json"
            metadata = {
                "analysis_type": "layout",
                "method_type": "article_reconstruction",
                "model_name": "headline_based_grouping",
                "model_version": None,
                "source": source_name,
                "created_at": datetime.now().isoformat(),
                "parameters": {
                    "vertical_threshold": self.vertical_threshold,
                    "horizontal_threshold": self.horizontal_threshold,
                    "min_text_length": self.min_text_length,
                    **(metadata_params or {}),
                },
                "total_articles": len(articles),
                "output_format": format,
            }
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {metadata_file}")
