"""
api/index.py
============
Vercel serverless entry point — imports the FastAPI app from server.py.
Vercel looks for `app` in api/index.py by convention.
"""

from api.server import app  # noqa: F401 — Vercel picks up `app` automatically
