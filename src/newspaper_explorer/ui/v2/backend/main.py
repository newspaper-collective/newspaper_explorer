"""
FastAPI backend for Historical Newspaper Explorer

A modern REST API for exploring historical newspaper collections.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.sources import list_available_sources

from .routers import (
    sources,
    data,
    results,
    entities,
    keywords,
    layout,
    topics,
    emotions,
    concepts,
    images,
    search,
)

# Create FastAPI app
app = FastAPI(
    title="Historical Newspaper Explorer API",
    description="REST API for exploring historical newspaper collections",
    version="2.0.0",
)

# Configure CORS for Vue frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "http://172.20.3.10:7860",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://172.20.3.10:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sources.router, prefix="/api/sources", tags=["sources"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(entities.router, prefix="/api/entities", tags=["entities"])
app.include_router(keywords.router, prefix="/api/keywords", tags=["keywords"])
app.include_router(layout.router, prefix="/api/layout", tags=["layout"])
app.include_router(topics.router, prefix="/api/topics", tags=["topics"])
app.include_router(emotions.router, prefix="/api/emotions", tags=["emotions"])
app.include_router(concepts.router, prefix="/api/concepts", tags=["concepts"])
app.include_router(images.router, prefix="/api/images", tags=["images"])
app.include_router(search.router, prefix="/api/search", tags=["search"])

# Mount static files for images
config = get_config()
for source in list_available_sources():
    images_path = Path(config.data_dir) / "raw" / source / "images"
    if images_path.exists():
        app.mount(
            f"/static/{source}/images",
            StaticFiles(directory=str(images_path)),
            name=f"{source}_images",
        )


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Historical Newspaper Explorer API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
