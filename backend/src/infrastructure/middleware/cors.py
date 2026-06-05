"""CORS middleware configuration.

Cho phép frontend client (Vite dev server) gọi tới FastAPI server (port 8000)
mà không bị browser chặn.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def register_cors(app: FastAPI) -> None:
    """Thêm CORS middleware vào FastAPI app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",   # Vite dev server
            "http://127.0.0.1:5173",
            "http://localhost:3000",   
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
