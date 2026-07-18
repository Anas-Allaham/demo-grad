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

# Non-destructive additive migrations. Each entry is a column that newer code
# needs; `migrate()` adds any that a pre-existing database is missing via
# PRAGMA table_info + ALTER TABLE, so an old app.db keeps all its data.
COLUMN_MIGRATIONS: Dict[str, List[tuple]] = {
    # Separate the concepts the old schema merged under a single `distance`.
    "attempt_phoneme_events": [
        ("articulatory_distance", "REAL"),  # raw PanPhon feature distance [0,1]
        ("alignment_cost", "REAL"),         # DP alignment cost (NOT a distance)
    ],
    # occurrence_count (phoneme occurrences) is tracked separately from
    # attempts_count, which now means independent recordings.
    "phoneme_skill_state": [
        ("occurrence_count", "INTEGER NOT NULL DEFAULT 0"),
    ],
    # Richer per-attempt bookkeeping + audio-quality gating.
    "attempts": [
        ("raw_weighted_per", "REAL"),
        ("quality_weight", "REAL"),
        ("scorable", "INTEGER NOT NULL DEFAULT 1"),
        ("rejected_reason", "TEXT"),
    ],
}


_PRIOR = 1.0


def migrate(conn: sqlite3.Connection) -> None:
    """Add any columns newer code needs that an existing DB lacks, then
    canonicalize legacy phoneme keys. Purely additive / evidence-preserving:
    never drops or destroys user data."""
    for table, columns in COLUMN_MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()
    _canonicalize_phoneme_states(conn)


def _canonicalize_phoneme_states(conn: sqlite3.Connection) -> None:
    """One-time, idempotent, non-destructive cleanup: fold legacy
    non-canonical phoneme_skill_state keys (e.g. 'r', 'iː', 'ɔr') into their
    canonical form ('ɹ', 'i', ...), MERGING their Beta evidence rather than
    dropping any. A no-op once every key is already canonical."""
    from phoneme_vectors_professional import canonicalize_phoneme

    rows = conn.execute(
        "SELECT user_id, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at "
        "FROM phoneme_skill_state"
    ).fetchall()

    # Nothing to do if every stored key is already canonical and unique.
    needs_work = False
    seen = set()
    for r in rows:
        canon = canonicalize_phoneme(r["phoneme"])
        key = (r["user_id"], canon)
        if canon != r["phoneme"] or key in seen:
            needs_work = True
            break
        seen.add(key)
    if not needs_work:
        return

    merged: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        canon = canonicalize_phoneme(r["phoneme"])
        if not canon:
            continue
        key = (r["user_id"], canon)
        acc = merged.get(key)
        if acc is None:
            merged[key] = {
                "alpha": r["alpha"],
                "beta": r["beta"],
                "attempts_count": r["attempts_count"] or 0,
                "occurrence_count": (r["occurrence_count"] or 0),
                "last_practiced_at": r["last_practiced_at"],
            }
        else:
            # Combine evidence above the shared prior so priors don't stack.
            acc["alpha"] += (r["alpha"] - _PRIOR)
            acc["beta"] += (r["beta"] - _PRIOR)
            acc["attempts_count"] += (r["attempts_count"] or 0)
            acc["occurrence_count"] += (r["occurrence_count"] or 0)
            if (r["last_practiced_at"] or "") > (acc["last_practiced_at"] or ""):
                acc["last_practiced_at"] = r["last_practiced_at"]

    conn.execute("DELETE FROM phoneme_skill_state")
    conn.executemany(
        """INSERT INTO phoneme_skill_state
               (user_id, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (uid, ph, v["alpha"], v["beta"], v["attempts_count"], v["occurrence_count"], v["last_practiced_at"])
            for (uid, ph), v in merged.items()
        ],
    )
    conn.commit()


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
    migrate(conn)


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
    raw_weighted_per: Optional[float] = None,
    quality_weight: Optional[float] = None,
    scorable: bool = True,
    rejected_reason: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    conn = conn or get_connection()
    cur = conn.execute(
        """INSERT INTO attempts
           (user_id, exercise_id, text, reference_ipa, predicted_ipa,
            phoneme_error_rate, weighted_error, raw_weighted_per,
            quality_weight, scorable, rejected_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, exercise_id, text, reference_ipa, predicted_ipa,
            phoneme_error_rate, weighted_error, raw_weighted_per,
            quality_weight, 1 if scorable else 0, rejected_reason,
        ),
    )
    conn.commit()
    return cur.lastrowid


def record_phoneme_events(
    attempt_id: int,
    alignment: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    rows = []
    for position, row in enumerate(alignment):
        art = row.get("articulatory_distance", row.get("distance"))
        cost = row.get("alignment_cost")
        # `distance` stays NOT NULL for backward compatibility; fall back to
        # the alignment cost for gap rows that have no articulatory distance.
        compat_distance = art if art is not None else (cost if cost is not None else 0.0)
        rows.append((
            attempt_id,
            position,
            None if row.get("expected") in (None, "-") else row["expected"],
            None if row.get("spoken") in (None, "-") else row["spoken"],
            row.get("result"),
            float(compat_distance),
            None if art is None else float(art),
            None if cost is None else float(cost),
        ))
    conn.executemany(
        """INSERT INTO attempt_phoneme_events
           (attempt_id, position, expected_phoneme, spoken_phoneme, operation,
            distance, articulatory_distance, alignment_cost)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
    occurrence_count: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Persist one phoneme's mastery state. ``attempts_count`` now means the
    number of independent recordings; ``occurrence_count`` is the number of
    phoneme occurrences observed."""
    conn = conn or get_connection()
    conn.execute(
        """INSERT INTO phoneme_skill_state
               (user_id, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_id, phoneme) DO UPDATE SET
               alpha=excluded.alpha,
               beta=excluded.beta,
               attempts_count=excluded.attempts_count,
               occurrence_count=excluded.occurrence_count,
               last_practiced_at=excluded.last_practiced_at""",
        (user_id, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at),
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


def update_sentence_tags(
    sentence_id: int,
    reference_ipa: str,
    phoneme_counts: Dict[str, int],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Re-tag an existing bank sentence in place: refresh its reference IPA and
    replace its phoneme tags. Content-only -- never touches user history
    (attempts / mastery), so it is safe to re-canonicalize an old bank."""
    conn = conn or get_connection()
    conn.execute(
        "UPDATE exercise_bank SET reference_ipa = ? WHERE id = ?",
        (reference_ipa, sentence_id),
    )
    conn.execute("DELETE FROM sentence_phonemes WHERE sentence_id = ?", (sentence_id,))
    conn.executemany(
        "INSERT INTO sentence_phonemes (sentence_id, phoneme, count) VALUES (?, ?, ?)",
        [(sentence_id, phoneme, count) for phoneme, count in phoneme_counts.items()],
    )
    conn.commit()


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


def count_exercise_bank(conn: Optional[sqlite3.Connection] = None) -> int:
    conn = conn or get_connection()
    row = conn.execute("SELECT COUNT(*) AS n FROM exercise_bank").fetchone()
    return int(row["n"]) if row else 0


def get_all_bank_phonemes(conn: Optional[sqlite3.Connection] = None) -> List[str]:
    """Distinct phonemes currently tagged into the exercise bank -- used by the
    startup inventory validation and /health."""
    conn = conn or get_connection()
    return [row["phoneme"] for row in conn.execute(
        "SELECT DISTINCT phoneme FROM sentence_phonemes"
    ).fetchall()]


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


# -----------------------------
# Evidence / confusion aggregation for assessment + confusion-aware exercises
# -----------------------------
def _scorable_filter() -> str:
    """SQL fragment: only count attempts that passed the audio-quality gate.
    Tolerates pre-migration rows where `scorable` is NULL."""
    return "(a.scorable IS NULL OR a.scorable != 0)"


def get_phoneme_context_stats(
    user_id: int, conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Dict[str, int]]:
    """Per expected phoneme: how many independent recordings it appeared in
    and across how many distinct prompt texts. Used by evidence-aware level
    eligibility (>= N recordings across >= M prompts)."""
    conn = conn or get_connection()
    rows = conn.execute(
        f"""SELECT e.expected_phoneme AS phoneme,
                   COUNT(DISTINCT a.id) AS recordings,
                   COUNT(DISTINCT a.text) AS distinct_prompts,
                   COUNT(*) AS occurrences
            FROM attempt_phoneme_events e
            JOIN attempts a ON e.attempt_id = a.id
            WHERE a.user_id = ?
              AND e.expected_phoneme IS NOT NULL
              AND {_scorable_filter()}
            GROUP BY e.expected_phoneme""",
        (user_id,),
    ).fetchall()
    return {
        row["phoneme"]: {
            "recordings": row["recordings"],
            "distinct_prompts": row["distinct_prompts"],
            "occurrences": row["occurrences"],
        }
        for row in rows
    }


def get_confusion_pairs(
    user_id: int, limit: Optional[int] = None, conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """Most frequent expected->spoken substitution pairs for a user, ordered
    by frequency. Feeds confusion-aware exercise selection (e.g. θ->s)."""
    conn = conn or get_connection()
    query = f"""SELECT e.expected_phoneme AS expected,
                       e.spoken_phoneme AS spoken,
                       COUNT(*) AS count
                FROM attempt_phoneme_events e
                JOIN attempts a ON e.attempt_id = a.id
                WHERE a.user_id = ?
                  AND e.operation LIKE '%substitution%'
                  AND e.expected_phoneme IS NOT NULL
                  AND e.spoken_phoneme IS NOT NULL
                  AND {_scorable_filter()}
                GROUP BY e.expected_phoneme, e.spoken_phoneme
                ORDER BY count DESC"""
    params: tuple = (user_id,)
    if limit is not None:
        query += " LIMIT ?"
        params = (user_id, limit)
    rows = conn.execute(query, params).fetchall()
    return [
        {"expected": row["expected"], "spoken": row["spoken"], "count": row["count"]}
        for row in rows
    ]


def get_scorable_recording_count(user_id: int, conn: Optional[sqlite3.Connection] = None) -> int:
    """Number of the user's recordings that passed the audio-quality gate."""
    conn = conn or get_connection()
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM attempts a WHERE a.user_id = ? AND {_scorable_filter()}",
        (user_id,),
    ).fetchone()
    return int(row["n"]) if row else 0
