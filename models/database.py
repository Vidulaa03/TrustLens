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
        conn.commit()


def insert_analysis(record):
    with closing(get_connection()) as conn:
        conn.execute(
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
