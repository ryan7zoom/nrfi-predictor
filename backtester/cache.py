"""SQLite cache: (namespace, key, as_of_date) -> JSON value."""

import json
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (namespace, key, as_of_date)
        )
    """)
    return conn


def get(namespace: str, key: str, as_of_date: str):
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT value FROM cache WHERE namespace=? AND key=? AND as_of_date=?",
            (namespace, key, as_of_date),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    finally:
        conn.close()


def set(namespace: str, key: str, as_of_date: str, value) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache (namespace, key, as_of_date, value) VALUES (?, ?, ?, ?)",
            (namespace, key, as_of_date, json.dumps(value)),
        )
        conn.commit()
    finally:
        conn.close()


def get_or_compute(namespace: str, key: str, as_of_date: str, compute_fn):
    cached = get(namespace, key, as_of_date)
    if cached is not None:
        return cached
    value = compute_fn()
    set(namespace, key, as_of_date, value)
    return value
