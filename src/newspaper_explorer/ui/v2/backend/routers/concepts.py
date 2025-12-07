"""
Concept extraction and knowledge graph endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import date
import polars as pl
from pathlib import Path

from newspaper_explorer.config.base import get_config
from newspaper_explorer.models.api.concepts import Concept, ConceptRelation

router = APIRouter()


@router.get("/{source_name}/", response_model=List[Concept])
async def get_concepts(
    source_name: str,
    category: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Get list of concepts with optional filtering"""
    try:
        config = get_config()
        concepts_path = Path(config.results_dir) / source_name / "concepts" / "concepts.parquet"

        if not concepts_path.exists():
            raise HTTPException(status_code=404, detail="No concept data available")

        df = pl.read_parquet(concepts_path)

        # Apply filters
        if category and "category" in df.columns:
            df = df.filter(df["category"] == category)
        if start_date and "date" in df.columns:
            df = df.filter(df["date"] >= start_date)
        if end_date and "date" in df.columns:
            df = df.filter(df["date"] <= end_date)

        # Aggregate by concept
        concepts_df = (
            df.group_by("concept")
            .agg(
                [
                    pl.count().alias("frequency"),
                    (
                        df["category"].first().alias("category")
                        if "category" in df.columns
                        else pl.lit(None).alias("category")
                    ),
                ]
            )
            .sort("frequency", descending=True)
            .head(limit)
        )

        # Convert to response model
        concepts = []
        for row in concepts_df.iter_rows(named=True):
            concepts.append(
                Concept(
                    concept=row["concept"],
                    frequency=row["frequency"],
                    category=row.get("category"),
                )
            )

        return concepts
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No concept data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/relations", response_model=List[ConceptRelation])
async def get_concept_relations(
    source_name: str,
    concept: Optional[str] = None,
    min_weight: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Get concept relationships for knowledge graph"""
    try:
        config = get_config()
        relations_path = Path(config.results_dir) / source_name / "concepts" / "relations.parquet"

        if not relations_path.exists():
            raise HTTPException(status_code=404, detail="No concept relations available")

        df = pl.read_parquet(relations_path)

        # Apply filters
        if concept:
            df = df.filter((df["source"] == concept) | (df["target"] == concept))

        df = df.filter(df["weight"] >= min_weight)

        # Sort by weight and limit
        df = df.sort("weight", descending=True).head(limit)

        # Convert to response model
        relations = []
        for row in df.iter_rows(named=True):
            relations.append(
                ConceptRelation(
                    source=row["source"],
                    target=row["target"],
                    weight=row["weight"],
                    relation_type=row.get("relation_type"),
                )
            )

        return relations
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No concept relations available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/categories", response_model=List[str])
async def get_concept_categories(source_name: str):
    """Get list of unique concept categories"""
    try:
        config = get_config()
        concepts_path = Path(config.results_dir) / source_name / "concepts" / "concepts.parquet"

        if not concepts_path.exists():
            return []

        df = pl.read_parquet(concepts_path)
        if "category" not in df.columns:
            return []

        categories = df["category"].drop_nulls().unique().to_list()
        return sorted(categories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
