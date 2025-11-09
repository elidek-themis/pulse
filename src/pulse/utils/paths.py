from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent

DATA = ROOT / "data"
TASKS = DATA / "tasks"
RESULTS = DATA / "results"

DOCS = DATA / "docs"
COMPLETIONS = DOCS / "completions"
PERSONAS = DOCS / "personas"

# import os
# import shutil
# import tempfile
# from pathlib import Path

# ROOT = Path(__file__).parent.parent.parent.parent

# # Predefined data directory (from git repo)
# DATA = ROOT / "data"
# PREDEFINED_TASKS = DATA / "tasks"
# PREDEFINED_RESULTS = DATA / "results"
# PREDEFINED_DOCS = DATA / "docs"
# PREDEFINED_COMPLETIONS = PREDEFINED_DOCS / "completions"
# PREDEFINED_PERSONAS = PREDEFINED_DOCS / "personas"

# # Check if session isolation is enabled
# USE_SESSION_STORAGE = os.getenv("USE_SESSION_STORAGE", "false").lower() == "true"


# def get_session_dir() -> Path:
#     """Get or create a session-specific directory with predefined data."""
#     import streamlit as st
#     from streamlit.runtime.scriptrunner import get_script_run_ctx

#     # Get unique session ID
#     ctx = get_script_run_ctx()
#     session_id = ctx.session_id if ctx else "default"

#     # Create session-specific directory
#     session_dir = Path(tempfile.gettempdir()) / "pulse_sessions" / session_id

#     # Only copy predefined data if this is a new session
#     if not session_dir.exists():
#         session_dir.mkdir(parents=True, exist_ok=True)
#         _copy_predefined_data(session_dir)

#     return session_dir


# def _copy_predefined_data(session_dir: Path) -> None:
#     """Copy predefined data from repo to session directory."""
#     mappings = {
#         PREDEFINED_TASKS: session_dir / "tasks",
#         PREDEFINED_RESULTS: session_dir / "results",
#         PREDEFINED_PERSONAS: session_dir / "docs" / "personas",
#         PREDEFINED_COMPLETIONS: session_dir / "docs" / "completions",
#     }

#     for source, dest in mappings.items():
#         if source.exists():
#             shutil.copytree(source, dest, dirs_exist_ok=True)
#         else:
#             dest.mkdir(parents=True, exist_ok=True)


# # Choose storage mode based on environment variable
# if USE_SESSION_STORAGE:
#     # Session-isolated storage (for HF Spaces)
#     SESSION_DIR = get_session_dir()
#     TASKS = SESSION_DIR / "tasks"
#     RESULTS = SESSION_DIR / "results"
#     DOCS = SESSION_DIR / "docs"
#     COMPLETIONS = DOCS / "completions"
#     PERSONAS = DOCS / "personas"
# else:
#     # Shared storage (for local development)
#     TASKS = DATA / "tasks"
#     RESULTS = DATA / "results"
#     DOCS = DATA / "docs"
#     COMPLETIONS = DOCS / "completions"
#     PERSONAS = DOCS / "personas"

# .env
# USE_SESSION_STORAGE=false

# ---
# title: PULSE
# emoji: 📊
# colorFrom: blue
# colorTo: green
# sdk: streamlit
# sdk_version: 1.40.0
# app_file: src/pulse/app.py
# pinned: false
# env:
#   - USE_SESSION_STORAGE=true
# ---

# FROM python:3.11-slim

# WORKDIR /app

# # Copy project files
# COPY . .

# # Install dependencies
# RUN pip install -r requirements.txt

# # Enable session storage for multi-user deployment
# ENV USE_SESSION_STORAGE=true

# EXPOSE 8501

# CMD ["streamlit", "run", "src/pulse/app.py"]

# # Local (shared storage)
# streamlit run src/pulse/app.py

# # Production (session-isolated)
# USE_SESSION_STORAGE=true streamlit run src/pulse/app.py

# # Docker
# docker run -e USE_SESSION_STORAGE=true pulse-app
