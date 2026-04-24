"""
DocSage App — Entry point for the Streamlit application.

HOW TO RUN:
    cd LangCGS/src/RealWorldProjects
    streamlit run docsage/app.py

    Or from the project root:
    streamlit run LangCGS/src/RealWorldProjects/docsage/app.py

WHY THIS FILE IS SEPARATE FROM main.py:
    app.py is the thin entry point that Streamlit runs.
    main.py contains the actual orchestration logic.
    This separation follows the same pattern as the reference project
    (langgraphagenticai) and keeps the entry point minimal.
"""

import sys
import os
import logging

# Add the RealWorldProjects directory to Python path so that
# "from docsage.xxx import yyy" works when Streamlit runs this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from the project root .env file
from dotenv import load_dotenv

# Try multiple possible .env locations (handles different working directories)
_app_dir = os.path.dirname(os.path.abspath(__file__))
_env_paths = [
    os.path.join(_app_dir, "..", "..", "..", "..", ".env"),  # From docsage/app.py -> LangCGS/.env
    os.path.join(_app_dir, "..", "..", "..", ".env"),        # Fallback
    os.path.join(os.getcwd(), ".env"),                       # Current working directory
    os.path.join(os.getcwd(), "..", "..", ".env"),            # Up from RealWorldProjects
]
for _env_path in _env_paths:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy sub-library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from docsage.main import run_docsage

if __name__ == "__main__":
    run_docsage()
