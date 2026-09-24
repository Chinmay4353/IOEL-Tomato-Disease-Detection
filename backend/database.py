"""SQLite database helper for the tomato detector backend."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "predictions.db"


def get_connection() -> sqlite3.Connection:
    """Create or connect to the SQLite database."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Initialize the statistics table used by the backend."""
    connection = get_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
    finally:
        connection.close()
