#!/usr/bin/env python3
"""Startup script - reads PORT from environment for Railway compatibility."""
import os
import uvicorn

port = int(os.environ.get("PORT", 8000))
uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")
