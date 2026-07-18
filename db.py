"""
SQLite persistence for user profiles, attempt history, and the practice
exercise bank.

Kept dependency-free (stdlib `sqlite3` only) and importable outside Flask
(e.g. from `scripts/build_exercise_bank.py`) — nothing here reads from
`flask.g` or any request context.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"

_connection: Optional[sqlite3.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exercise_bank (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    reference_ipa TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    level_proxy REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'retrieval',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sentence_phonemes (
    sentence_id INTEGER NOT NULL REFERENCES exercise_bank(id),
    phoneme TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (sentence_id, phoneme)
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    exercise_id INTEGER REFERENCES exercise_bank(id),
    text TEXT NOT NULL,
    reference_ipa TEXT NOT NULL,
    predicted_ipa TEXT NOT NULL,
    phoneme_error_rate REAL NOT NULL,
    weighted_error REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS attempt_phoneme_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL REFERENCES attempts(id),
    position INTEGER NOT NULL,
    expected_phoneme TEXT,
    spoken_phoneme TEXT,
    operation TEXT NOT NULL,
    distance REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS phoneme_skill_state (
    user_id INTEGER NOT NULL REFERENCES users(id),
    phoneme TEXT NOT NULL,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    attempts_count INTEGER NOT NULL DEFAULT 0,
    last_practiced_at TEXT,
    PRIMARY KEY (user_id, phoneme)
);

CREATE TABLE IF NOT EXISTS practice_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    exercise_id INTEGER NOT NULL REFERENCES exercise_bank(id),
    target_phonemes TEXT NOT NULL,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_attempt_id INTEGER REFERENCES attempts(id)
);

CREATE INDEX IF NOT EXISTS idx_attempt_phoneme_events_attempt
    ON attempt_phoneme_events(attempt_id);
CREATE INDEX IF NOT EXISTS idx_practice_assignments_user
    ON practice_assignments(user_id, assigned_at);
"""


def get_connection() -> sqlite3.Connection:
    """Lazily-opened, process-wide connection. WAL mode keeps a single
    local writer safe for this app's scale (one user, a local demo)."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA foreign_keys=ON")
    return _connection


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    conn = conn or get_connection()
    conn.executescript(SCHEMA)
    conn.commit()


# -----------------------------
# Users — no auth, just a named profile. get_or_create_user is the only
# entry point most callers need; a name typed into the UI either resolves
# to its existing history or starts a fresh one.
# -----------------------------
def get_or_create_user(name: str, conn: Optional[sqlite3.Connection] = None) -> sqlite3.Row:
    conn = conn or get_connection()
    name = name.strip()
    existing = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
    if existing is not None:
        return existing
    conn.execute("INSERT INTO users (name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()


def get_user_by_name(name: str, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute("SELECT * FROM users WHERE name = ?", (name.strip(),)).fetchone()


def get_user_by_id(user_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def list_users(conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute("SELECT id, name, created_at FROM users ORDER BY name COLLATE NOCASE").fetchall()


# -----------------------------
# Attempts + raw phoneme events
# -----------------------------
def record_attempt(
    user_id: int,
    text: str,
    reference_ipa: str,
    predicted_ipa: str,
    phoneme_error_rate: float,
    weighted_error: float,
    exercise_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    conn = conn or get_connection()
    cur = conn.execute(
        """INSERT INTO attempts
           (user_id, exercise_id, text, reference_ipa, predicted_ipa, phoneme_error_rate, weighted_error)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, exercise_id, text, reference_ipa, predicted_ipa, phoneme_error_rate, weighted_error),
    )
    conn.commit()
    return cur.lastrowid


def record_phoneme_events(
    attempt_id: int,
    alignment: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    rows = [
        (
            attempt_id,
            position,
            None if row.get("expected") in (None, "-") else row["expected"],
            None if row.get("spoken") in (None, "-") else row["spoken"],
            row.get("result"),
            float(row.get("distance", 0.0)),
        )
        for position, row in enumerate(alignment)
    ]
    conn.executemany(
        """INSERT INTO attempt_phoneme_events
           (attempt_id, position, expected_phoneme, spoken_phoneme, operation, distance)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def get_attempt_phoneme_events(attempt_id: int, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM attempt_phoneme_events WHERE attempt_id = ? ORDER BY position",
        (attempt_id,),
    ).fetchall()


def get_user_attempts(user_id: int, limit: int = 20, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM attempts WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()


# -----------------------------
# Phoneme skill state (mastery cache)
# -----------------------------
def get_phoneme_state(user_id: int, phoneme: str, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM phoneme_skill_state WHERE user_id = ? AND phoneme = ?",
        (user_id, phoneme),
    ).fetchone()


def get_all_phoneme_states(user_id: int, conn: Optional[sqlite3.Connection] = None) -> List[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute(
        "SELECT * FROM phoneme_skill_state WHERE user_id = ?", (user_id,)
    ).fetchall()


def upsert_phoneme_state(
    user_id: int,
    phoneme: str,
    alpha: float,
    beta: float,
    attempts_count: int,
    last_practiced_at: str,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    conn.execute(
        """INSERT INTO phoneme_skill_state (user_id, phoneme, alpha, beta, attempts_count, last_practiced_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_id, phoneme) DO UPDATE SET
               alpha=excluded.alpha,
               beta=excluded.beta,
               attempts_count=excluded.attempts_count,
               last_practiced_at=excluded.last_practiced_at""",
        (user_id, phoneme, alpha, beta, attempts_count, last_practiced_at),
    )
    conn.commit()


# -----------------------------
# Exercise bank
# -----------------------------
def insert_sentence(
    text: str,
    reference_ipa: str,
    word_count: int,
    level_proxy: float,
    phoneme_counts: Dict[str, int],
    source: str = "retrieval",
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[int]:
    """Insert a sentence + its phoneme tags. Returns None (no-op) if the
    exact sentence text is already in the bank."""
    conn = conn or get_connection()
    existing = conn.execute("SELECT id FROM exercise_bank WHERE text = ?", (text,)).fetchone()
    if existing:
        return None
    cur = conn.execute(
        """INSERT INTO exercise_bank (text, reference_ipa, word_count, level_proxy, source)
           VALUES (?, ?, ?, ?, ?)""",
        (text, reference_ipa, word_count, level_proxy, source),
    )
    sentence_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO sentence_phonemes (sentence_id, phoneme, count) VALUES (?, ?, ?)",
        [(sentence_id, phoneme, count) for phoneme, count in phoneme_counts.items()],
    )
    conn.commit()
    return sentence_id


def get_sentence_by_id(sentence_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute("SELECT * FROM exercise_bank WHERE id = ?", (sentence_id,)).fetchone()


def get_sentence_phonemes(sentence_id: int, conn: Optional[sqlite3.Connection] = None) -> Dict[str, int]:
    conn = conn or get_connection()
    rows = conn.execute(
        "SELECT phoneme, count FROM sentence_phonemes WHERE sentence_id = ?", (sentence_id,)
    ).fetchall()
    return {row["phoneme"]: row["count"] for row in rows}


def get_sentences_covering_any(phonemes: Iterable[str], conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """Return bank sentences that contain at least one of `phonemes`, each
    annotated with its full phoneme-count map so callers can score overlap."""
    conn = conn or get_connection()
    phonemes = list(phonemes)
    if not phonemes:
        return []
    placeholders = ",".join("?" for _ in phonemes)
    sentence_ids = [
        row["sentence_id"]
        for row in conn.execute(
            f"SELECT DISTINCT sentence_id FROM sentence_phonemes WHERE phoneme IN ({placeholders})",
            phonemes,
        ).fetchall()
    ]
    return [_load_sentence_with_phonemes(sid, conn) for sid in sentence_ids]


def get_all_sentences(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    conn = conn or get_connection()
    ids = [row["id"] for row in conn.execute("SELECT id FROM exercise_bank").fetchall()]
    return [_load_sentence_with_phonemes(sid, conn) for sid in ids]


def _load_sentence_with_phonemes(sentence_id: int, conn: sqlite3.Connection) -> Dict[str, Any]:
    sentence = get_sentence_by_id(sentence_id, conn)
    phoneme_counts = get_sentence_phonemes(sentence_id, conn)
    return {
        "id": sentence["id"],
        "text": sentence["text"],
        "reference_ipa": sentence["reference_ipa"],
        "word_count": sentence["word_count"],
        "level_proxy": sentence["level_proxy"],
        "source": sentence["source"],
        "phoneme_counts": phoneme_counts,
    }


# -----------------------------
# Practice assignments (what was served, and how it scored)
# -----------------------------
def record_practice_assignment(
    user_id: int,
    exercise_id: int,
    target_phonemes: List[str],
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    conn = conn or get_connection()
    cur = conn.execute(
        "INSERT INTO practice_assignments (user_id, exercise_id, target_phonemes) VALUES (?, ?, ?)",
        (user_id, exercise_id, json.dumps(target_phonemes)),
    )
    conn.commit()
    return cur.lastrowid


def complete_latest_assignment(
    user_id: int,
    exercise_id: int,
    attempt_id: int,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Stamp the most recent open (uncompleted) assignment for this
    (user, sentence) pair with the resulting attempt."""
    conn = conn or get_connection()
    row = conn.execute(
        """SELECT id FROM practice_assignments
           WHERE user_id = ? AND exercise_id = ? AND completed_attempt_id IS NULL
           ORDER BY assigned_at DESC LIMIT 1""",
        (user_id, exercise_id),
    ).fetchone()
    if row is None:
        return
    conn.execute(
        "UPDATE practice_assignments SET completed_attempt_id = ? WHERE id = ?",
        (attempt_id, row["id"]),
    )
    conn.commit()


def get_recently_served_sentence_ids(user_id: int, limit: int = 15, conn: Optional[sqlite3.Connection] = None) -> set:
    conn = conn or get_connection()
    rows = conn.execute(
        """SELECT exercise_id FROM practice_assignments
           WHERE user_id = ? ORDER BY assigned_at DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    return {row["exercise_id"] for row in rows}
