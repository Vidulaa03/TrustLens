import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "trustlens.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_connection()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_text TEXT NOT NULL,
                headline TEXT,
                timestamp TEXT NOT NULL,
                trust_score INTEGER,
                fake_probability REAL,
                real_probability REAL,
                bias_score REAL,
                ai_probability REAL,
                topic TEXT,
                headline_score INTEGER,
                risk_level TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                report_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


def recalculate_risk_levels():
    """Bring saved records in line with the current, corroborated risk rules."""
    from services.enhanced_analytics import calculate_risk_level

    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT id, trust_score, bias_score, headline_score, ai_probability FROM analyses"
        ).fetchall()
        for row in rows:
            risk_level = calculate_risk_level(
                row["trust_score"] or 0,
                row["bias_score"] or 0,
                0,  # Historic records do not retain the clickbait component.
                row["headline_score"] if row["headline_score"] is not None else 50,
                row["ai_probability"] or 0,
            )
            conn.execute("UPDATE analyses SET risk_level = ? WHERE id = ?", (risk_level, row["id"]))
        conn.commit()


def insert_analysis(record):
    with closing(get_connection()) as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses (
                article_text, headline, timestamp, trust_score, fake_probability,
                real_probability, bias_score, ai_probability, topic, headline_score, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["article_text"],
                record.get("headline"),
                record["timestamp"],
                record["trust_score"],
                record.get("fake_probability"),
                record.get("real_probability"),
                record.get("bias_score"),
                record.get("ai_probability"),
                record.get("topic"),
                record.get("headline_score"),
                record.get("risk_level"),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def fetch_all_analyses():
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM analyses
            ORDER BY datetime(timestamp) DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def fetch_recent(limit=10):
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM analyses
            ORDER BY datetime(timestamp) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def delete_analysis(analysis_id):
    with closing(get_connection()) as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()


def clear_history():
    with closing(get_connection()) as conn:
        conn.execute("DELETE FROM analyses")
        conn.commit()


def create_user(name, username, email, password_hash):
    with closing(get_connection()) as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, username, email, password_hash) VALUES (?, ?, ?, ?)",
            (name, username, email, password_hash),
        )
        conn.commit()
        return cursor.lastrowid


def get_user_by_username_or_email(identifier):
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ? LIMIT 1",
            (identifier, identifier),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id):
    with closing(get_connection()) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ? LIMIT 1", (user_id,)).fetchone()
        return dict(row) if row else None


def save_report(analysis_id, user_id, report_name, notes=None):
    """Save a report by linking an analysis to a user with a custom name."""
    with closing(get_connection()) as conn:
        cursor = conn.execute(
            """
            INSERT INTO saved_reports (analysis_id, user_id, report_name, notes)
            VALUES (?, ?, ?, ?)
            """,
            (analysis_id, user_id, report_name, notes),
        )
        conn.commit()
        return cursor.lastrowid


def get_saved_reports_for_user(user_id):
    """Fetch all saved reports for a given user."""
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT sr.id, sr.analysis_id, sr.report_name, sr.created_at, sr.notes,
                   a.headline, a.topic, a.trust_score, a.risk_level
            FROM saved_reports sr
            JOIN analyses a ON sr.analysis_id = a.id
            WHERE sr.user_id = ?
            ORDER BY sr.created_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_saved_report(report_id, user_id):
    """Fetch a specific saved report if it belongs to the user."""
    with closing(get_connection()) as conn:
        row = conn.execute(
            """
            SELECT sr.id, sr.analysis_id, sr.report_name, sr.created_at, sr.notes,
                   a.*
            FROM saved_reports sr
            JOIN analyses a ON sr.analysis_id = a.id
            WHERE sr.id = ? AND sr.user_id = ?
            """,
            (report_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def delete_saved_report(report_id, user_id):
    """Delete a saved report if it belongs to the user."""
    with closing(get_connection()) as conn:
        # Check if report belongs to user first
        row = conn.execute(
            "SELECT id FROM saved_reports WHERE id = ? AND user_id = ?",
            (report_id, user_id),
        ).fetchone()
        if row:
            conn.execute("DELETE FROM saved_reports WHERE id = ?", (report_id,))
            conn.commit()
            return True
        return False

