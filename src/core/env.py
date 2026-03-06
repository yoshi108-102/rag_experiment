"""Environment bootstrap helpers."""

from __future__ import annotations

from dotenv import load_dotenv


def initialize_environment() -> None:
    """Load environment variables from `.env` into process env."""
    load_dotenv()
