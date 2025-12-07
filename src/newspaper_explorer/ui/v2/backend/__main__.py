"""
Run the FastAPI backend server

Usage:
    python -m newspaper_explorer.ui.v2.backend
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "newspaper_explorer.ui.v2.backend.main:app",
        host="127.0.0.1",
        port=8005,
        reload=True,
    )
