"""
SQLite cache: (namespace, key, as_of_date) -> JSON value.

Used from multiple threads when the backtest runs games concurrently.
SQLite serializes writes across connections, so concurrent access can
raise "database is locked" errors under load. Two things guard against
that: a longer busy timeout (SQLite will wait and retry internally
before giving up) and an explicit retry loop here as a second layer.
"""

import json
import sqlite3
import os
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache.db")

BUSY_TIMEOUT_SECONDS = 30
MAX_LOCK_RETRIES = 5


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=BUSY_TIMEOUT_SECONDS)
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


def _with_lock_retry(fn):
    last_exc = None
    for attempt in range(MAX_LOCK_RETRIES):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_exc = exc
            time.sleep(0.2 * (attempt + 1))
    raise last_exc


def get(namespace: str, key: str, as_of_date: str):
    def _do():
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

    return _with_lock_retry(_do)


def set(namespace: str, key: str, as_of_date: str, value) -> None:
    def _do():
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (namespace, key, as_of_date, value) VALUES (?, ?, ?, ?)",
                (namespace, key, as_of_date, json.dumps(value)),
            )
            conn.commit()
        finally:
            conn.close()

    _with_lock_retry(_do)


def get_or_compute(namespace: str, key: str, as_of_date: str, compute_fn):
    cached = get(namespace, key, as_of_date)
    if cached is not None:
        return cached
    value = compute_fn()
    set(namespace, key, as_of_date, value)
    return value
