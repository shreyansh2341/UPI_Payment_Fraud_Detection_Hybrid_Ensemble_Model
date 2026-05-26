"""
db.py — SQLite Persistence Layer for Fraud Detection History
═══════════════════════════════════════════════════════════
Lightweight storage for detection results and upload sessions.
Enables time-series analytics and dashboard history features.
"""
import sqlite3
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

DB_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DB_DIR / "fraud_history.db"


def _ensure_db_dir():
    DB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    """Thread-safe context manager for SQLite connections."""
    _ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database tables."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS upload_sessions (
                session_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                filename TEXT,
                dataset_type TEXT NOT NULL,
                total_count INTEGER DEFAULT 0,
                fraud_blocked INTEGER DEFAULT 0,
                novel_blocked INTEGER DEFAULT 0,
                review_flagged INTEGER DEFAULT 0,
                legitimate INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                transaction_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                amount REAL DEFAULT 0.0,
                transaction_type TEXT,
                decision TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                ae_score REAL DEFAULT 0.0,
                seq_score REAL DEFAULT 0.0,
                explanation TEXT,
                FOREIGN KEY (session_id) REFERENCES upload_sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_results_timestamp
                ON detection_results(timestamp);
            CREATE INDEX IF NOT EXISTS idx_results_decision
                ON detection_results(decision);
            CREATE INDEX IF NOT EXISTS idx_results_session
                ON detection_results(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_timestamp
                ON upload_sessions(timestamp);
        """)


def create_session(filename: str, dataset_type: str) -> str:
    """Create a new upload session. Returns session_id."""
    session_id = str(uuid.uuid4())[:12]
    timestamp = datetime.utcnow().isoformat()

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO upload_sessions
               (session_id, timestamp, filename, dataset_type)
               VALUES (?, ?, ?, ?)""",
            (session_id, timestamp, filename, dataset_type),
        )
    return session_id


def store_batch_results(session_id: str, results: list, amounts: list = None):
    """Store batch detection results and update session summary."""
    timestamp = datetime.utcnow().isoformat()

    counts = {"BLOCK": 0, "BLOCK_NOVEL": 0, "REVIEW": 0, "ALLOW": 0, "ERROR": 0}

    with get_connection() as conn:
        for i, r in enumerate(results):
            decision = r.get("decision_label", "ALLOW")
            counts[decision] = counts.get(decision, 0) + 1

            amt = float(amounts[i]) if amounts is not None and i < len(amounts) else 0.0

            conn.execute(
                """INSERT INTO detection_results
                   (session_id, transaction_index, timestamp, amount,
                    transaction_type, decision, confidence, ae_score,
                    seq_score, explanation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, i, timestamp, amt,
                    r.get("transaction_type", "paysim"),
                    decision,
                    r.get("score", 0.0),
                    r.get("ae_score", 0.0),
                    r.get("seq_score", 0.0),
                    r.get("explanation", ""),
                ),
            )

        conn.execute(
            """UPDATE upload_sessions SET
               total_count = ?,
               fraud_blocked = ?,
               novel_blocked = ?,
               review_flagged = ?,
               legitimate = ?,
               errors = ?
               WHERE session_id = ?""",
            (
                len(results),
                counts.get("BLOCK", 0),
                counts.get("BLOCK_NOVEL", 0),
                counts.get("REVIEW", 0),
                counts.get("ALLOW", 0),
                counts.get("ERROR", 0),
                session_id,
            ),
        )


def get_recent_detections(limit: int = 50) -> list:
    """Get most recent detection results."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT d.*, s.filename, s.dataset_type as ds_type
               FROM detection_results d
               JOIN upload_sessions s ON d.session_id = s.session_id
               ORDER BY d.timestamp DESC, d.id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_detection_history(days: int = 7) -> list:
    """Get detection results within a time window."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT d.*, s.filename
               FROM detection_results d
               JOIN upload_sessions s ON d.session_id = s.session_id
               WHERE d.timestamp >= ?
               ORDER BY d.timestamp ASC""",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_analytics(days: int = 7) -> dict:
    """Get aggregated fraud analytics for a time window."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    with get_connection() as conn:
        # Decision breakdown
        decision_counts = conn.execute(
            """SELECT decision, COUNT(*) as cnt
               FROM detection_results
               WHERE timestamp >= ?
               GROUP BY decision""",
            (cutoff,),
        ).fetchall()

        # Daily fraud series
        daily_series = conn.execute(
            """SELECT DATE(timestamp) as date,
                      decision,
                      COUNT(*) as cnt
               FROM detection_results
               WHERE timestamp >= ?
               GROUP BY DATE(timestamp), decision
               ORDER BY date ASC""",
            (cutoff,),
        ).fetchall()

        # Hourly heatmap
        hourly_heatmap = conn.execute(
            """SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                      CAST(strftime('%w', timestamp) AS INTEGER) as dow,
                      COUNT(*) as cnt
               FROM detection_results
               WHERE timestamp >= ? AND decision IN ('BLOCK', 'BLOCK_NOVEL', 'REVIEW')
               GROUP BY hour, dow""",
            (cutoff,),
        ).fetchall()

        # Score distributions
        score_dist = conn.execute(
            """SELECT ae_score, seq_score, confidence, decision
               FROM detection_results
               WHERE timestamp >= ?""",
            (cutoff,),
        ).fetchall()

        # Session summaries
        sessions = conn.execute(
            """SELECT *
               FROM upload_sessions
               WHERE timestamp >= ?
               ORDER BY timestamp DESC""",
            (cutoff,),
        ).fetchall()

        # Totals
        totals = conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as blocked,
                 SUM(CASE WHEN decision = 'BLOCK_NOVEL' THEN 1 ELSE 0 END) as novel,
                 SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) as review,
                 SUM(CASE WHEN decision = 'ALLOW' THEN 1 ELSE 0 END) as allowed,
                 AVG(confidence) as avg_confidence,
                 AVG(ae_score) as avg_ae_score,
                 AVG(seq_score) as avg_seq_score
               FROM detection_results
               WHERE timestamp >= ?""",
            (cutoff,),
        ).fetchone()

    return {
        "decision_counts": {r["decision"]: r["cnt"] for r in decision_counts},
        "daily_series": [dict(r) for r in daily_series],
        "hourly_heatmap": [dict(r) for r in hourly_heatmap],
        "score_distribution": [dict(r) for r in score_dist],
        "sessions": [dict(r) for r in sessions],
        "totals": dict(totals) if totals else {},
    }


def get_dashboard_summary() -> dict:
    """Get summary stats for the dashboard."""
    with get_connection() as conn:
        # All-time totals
        totals = conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as blocked,
                 SUM(CASE WHEN decision = 'BLOCK_NOVEL' THEN 1 ELSE 0 END) as novel,
                 SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) as review,
                 SUM(CASE WHEN decision = 'ALLOW' THEN 1 ELSE 0 END) as allowed
               FROM detection_results"""
        ).fetchone()

        # Last 24h
        cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        recent = conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN decision IN ('BLOCK', 'BLOCK_NOVEL') THEN 1 ELSE 0 END) as fraud
               FROM detection_results
               WHERE timestamp >= ?""",
            (cutoff_24h,),
        ).fetchone()

        # Last upload session
        last_session = conn.execute(
            """SELECT * FROM upload_sessions
               ORDER BY timestamp DESC LIMIT 1"""
        ).fetchone()

    return {
        "all_time": dict(totals) if totals else {},
        "last_24h": dict(recent) if recent else {},
        "last_session": dict(last_session) if last_session else None,
    }


# Initialize on import
init_db()
