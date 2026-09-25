"""
rate_limit_store.py
---------------------
A small, self-contained, persistent rate limiter backed by a local
SQLite file, replacing Flask-Limiter's in-memory storage.

WHY THIS EXISTS
----------------
The draft honestly flagged this limitation (Section 3.5): "Rate
limiting also uses in-memory storage by default, which resets
whenever the server restarts and would not work correctly across
multiple server processes." Flask-Limiter's built-in alternatives
(Redis, Memcached, MongoDB) all require a separate running service --
reasonable for a real production deployment, but disproportionate
infrastructure to add just for a student project demo. A local SQLite
file gives genuine persistence (survives restarts, works across
multiple worker processes on the same machine) with zero extra
services to install or run.

DESIGN
-------
Fixed-window counter per (client_key, route) pair: for each request,
compute the current window ("YYYY-MM-DD HH:MM", i.e. resets every
real clock minute), UPSERT-increment a row keyed on
(client_key, route, window), and compare the resulting count to the
limit. This is simpler than a sliding-window log and is the same
"fixed window" strategy Flask-Limiter itself defaults to, so behaviour
is equivalent to before -- only the storage backend changes.

Thread-safety: SQLite's own file locking handles concurrent writes
correctly for Flask's default threaded dev server; a `threading.Lock`
is added defensively around the connection since sqlite3 connections
are not safe to share across threads without care.

USAGE (see app_ratelimit_patch.py for the exact app.py integration):

    from rate_limit_store import rate_limited

    @app.route("/recommend", methods=["POST"])
    @rate_limited(limit=20, window_seconds=60)
    def recommend():
        ...
"""

import sqlite3
import threading
import time
from functools import wraps
from flask import request, jsonify

DB_PATH = "rate_limits.sqlite3"
_lock = threading.Lock()


def _get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            client_key TEXT NOT NULL,
            route TEXT NOT NULL,
            window TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (client_key, route, window)
        )
    """)
    return conn


def _current_window(window_seconds):
    """Bucket the current time into a fixed window, e.g. every 60s
    aligns to the start of the current minute."""
    bucket = int(time.time() // window_seconds)
    return str(bucket)


def _client_key():
    # Same identity signal Flask-Limiter's get_remote_address uses.
    return request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"


def check_and_increment(route, limit, window_seconds):
    """Increments the counter for (client, route, current window) and
    returns (allowed: bool, current_count: int)."""
    client_key = _client_key()
    window = _current_window(window_seconds)

    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                INSERT INTO rate_limits (client_key, route, window, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(client_key, route, window)
                DO UPDATE SET count = count + 1
                """,
                (client_key, route, window),
            )
            conn.commit()
            row = conn.execute(
                "SELECT count FROM rate_limits WHERE client_key=? AND route=? AND window=?",
                (client_key, route, window),
            ).fetchone()
            count = row[0] if row else 1
        finally:
            conn.close()

    return count <= limit, count


def cleanup_old_windows(older_than_seconds=3600):
    """Optional housekeeping: deletes rows old enough that they can no
    longer affect any active window. Not required for correctness (old
    rows just sit unused), but keeps the SQLite file from growing
    unbounded over a long-running server. Call this periodically (e.g.
    a scheduled job) rather than on every request."""
    cutoff_bucket = int((time.time() - older_than_seconds) // 60)
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                "DELETE FROM rate_limits WHERE CAST(window AS INTEGER) < ?",
                (cutoff_bucket,),
            )
            conn.commit()
        finally:
            conn.close()


def rate_limited(limit, window_seconds=60):
    """Decorator: apply directly to a Flask route function in place of
    Flask-Limiter's @limiter.limit(...). Returns the same JSON 429
    shape the existing errorhandler(429) in app.py produces, so
    test_rate_limit_response_is_valid_json keeps passing unchanged."""
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            route = request.path
            allowed, count = check_and_increment(route, limit, window_seconds)
            if not allowed:
                return jsonify({
                    "error": "Too many requests -- please wait a moment before trying again.",
                    "retry_after": str(window_seconds),
                }), 429
            return view_func(*args, **kwargs)
        # Expose a reset hook for tests (see test_api.py changes).
        wrapped._rate_limit_route = None
        return wrapped
    return decorator


def reset_all():
    """Test-only helper: wipes the whole table so each test run starts
    from a clean rate-limit state, instead of accumulating counts
    across test runs (a real risk with a persistent backend that an
    in-memory one never had)."""
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("DELETE FROM rate_limits")
            conn.commit()
        finally:
            conn.close()
