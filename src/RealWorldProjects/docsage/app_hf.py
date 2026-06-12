"""
DocSage App — Hugging Face Spaces entry point.

This file is used ONLY for Hugging Face deployment.
For local development, use app.py instead.

On HF Spaces, the docsage/ folder is at the repo root,
so sys.path points to the current directory.
API keys come from HF Secrets (Settings -> Repository secrets).
"""

import sys
import os
import logging

# On HF, the repo root contains the docsage/ package directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# On HF, secrets are injected as environment variables automatically.
# No .env file needed. But try loading one if it exists (for local testing).
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from docsage.main import run_docsage

run_docsage()
