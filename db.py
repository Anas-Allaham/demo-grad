"""
SQLite persistence for user profiles, attempt history, and the practice
exercise bank.

Kept dependency-free (stdlib `sqlite3` only) and importable outside Flask
(e.g. from `scripts/build_exercise_bank.py`) — nothing here reads from
`flask.g` or any request context.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("APP_DB_PATH", str(BASE_DIR / "app.db"))).expanduser()

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
        ("scoring_engine", "TEXT"),         # per-event engine provenance
    ],
    # occurrence_count (phoneme occurrences) is tracked separately from
    # attempts_count, which now means independent recordings.
    "phoneme_skill_state": [
        ("occurrence_count", "INTEGER NOT NULL DEFAULT 0"),
    ],
    # Richer per-attempt bookkeeping + audio-quality gating + scoring
    # provenance. scoring_trusted / mastery_updated default to 0 so that
    # PRE-EXISTING legacy rows are treated as UNTRUSTED unless explicitly
    # migrated -- assessment/confusion/diagnostic only count trusted rows.
    "attempts": [
        ("raw_weighted_per", "REAL"),
        ("quality_weight", "REAL"),
        ("scorable", "INTEGER NOT NULL DEFAULT 1"),
        ("rejected_reason", "TEXT"),
        ("scoring_engine", "TEXT"),
        ("scoring_trusted", "INTEGER NOT NULL DEFAULT 0"),
        ("mastery_updated", "INTEGER NOT NULL DEFAULT 0"),
        ("insertion_count", "INTEGER NOT NULL DEFAULT 0"),
        ("reference_unit_count", "INTEGER NOT NULL DEFAULT 0"),
        ("g2p_mode", "TEXT"),
        ("reference_g2p_trusted", "INTEGER NOT NULL DEFAULT 0"),
        ("reference_g2p_reason", "TEXT"),
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


def set_database_for_testing(path: "str | Path") -> None:
    """Point the module at a different SQLite file (tests use a temp DB so they
    never touch the real app.db). Closes any open connection first."""
    global _connection, DB_PATH
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass
    _connection = None
    DB_PATH = Path(path)


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
def _insert_attempt(
    conn: sqlite3.Connection,
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
    scoring_engine: Optional[str] = None,
    scoring_trusted: bool = False,
    mastery_updated: bool = False,
    insertion_count: int = 0,
    reference_unit_count: int = 0,
    g2p_mode: Optional[str] = None,
    reference_g2p_trusted: bool = False,
    reference_g2p_reason: Optional[str] = None,
) -> int:
    """Insert one attempt row WITHOUT committing (transaction-friendly)."""
    cur = conn.execute(
        """INSERT INTO attempts
           (user_id, exercise_id, text, reference_ipa, predicted_ipa,
            phoneme_error_rate, weighted_error, raw_weighted_per,
            quality_weight, scorable, rejected_reason,
            scoring_engine, scoring_trusted, mastery_updated, insertion_count,
            reference_unit_count, g2p_mode, reference_g2p_trusted, reference_g2p_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, exercise_id, text, reference_ipa, predicted_ipa,
            phoneme_error_rate, weighted_error, raw_weighted_per,
            quality_weight, 1 if scorable else 0, rejected_reason,
            scoring_engine, 1 if scoring_trusted else 0,
            1 if mastery_updated else 0, insertion_count, reference_unit_count,
            g2p_mode, 1 if reference_g2p_trusted else 0, reference_g2p_reason,
        ),
    )
    return cur.lastrowid


def record_attempt(conn: Optional[sqlite3.Connection] = None, **kwargs) -> int:
    """Insert one attempt and commit. Prefer ``record_recording_atomic`` for
    the full attempt+events+mastery+assignment write."""
    conn = conn or get_connection()
    attempt_id = _insert_attempt(conn, **kwargs)
    conn.commit()
    return attempt_id


def record_phoneme_events(
    attempt_id: int,
    alignment: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    conn = conn or get_connection()
    _insert_events(conn, attempt_id, alignment)
    conn.commit()


def _insert_events(
    conn: sqlite3.Connection,
    attempt_id: int,
    alignment: List[Dict[str, Any]],
    scoring_engine: Optional[str] = None,
) -> None:
    """Insert alignment events WITHOUT committing (transaction-friendly)."""
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
            scoring_engine,
        ))
    conn.executemany(
        """INSERT INTO attempt_phoneme_events
           (attempt_id, position, expected_phoneme, spoken_phoneme, operation,
            distance, articulatory_distance, alignment_cost, scoring_engine)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def record_recording_atomic(
    user_id: int,
    attempt_kwargs: Dict[str, Any],
    alignment: List[Dict[str, Any]],
    scoring_engine: Optional[str] = None,
    phoneme_states: Optional[Dict[str, Dict[str, Any]]] = None,
    complete_exercise_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Write attempt + events + mastery + assignment completion in ONE atomic
    SQLite transaction. If any step raises, the whole write rolls back so a
    partial failure can never leave inconsistent evidence (e.g. events without
    the mastery update, or an assignment marked complete for an attempt that
    was never fully recorded).

    ``phoneme_states``: {phoneme: {alpha, beta, attempts_count, occurrence_count,
    last_practiced_at}} to upsert (only when mastery was updated).
    ``complete_exercise_id``: mark the latest open assignment for this exercise
    complete (only when mastery was updated).
    """
    conn = conn or get_connection()
    with conn:  # BEGIN; commits on success, rolls back on any exception
        attempt_id = _insert_attempt(conn, user_id=user_id, **attempt_kwargs)
        _insert_events(conn, attempt_id, alignment, scoring_engine)
        if phoneme_states:
            for phoneme, st in phoneme_states.items():
                _upsert_phoneme_state(
                    conn, user_id, phoneme,
                    st["alpha"], st["beta"], st["attempts_count"],
                    st["occurrence_count"], st["last_practiced_at"],
                )
        if complete_exercise_id is not None:
            _complete_latest_assignment(conn, user_id, complete_exercise_id, attempt_id)
    return attempt_id


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
    _upsert_phoneme_state(conn, user_id, phoneme, alpha, beta, attempts_count, occurrence_count, last_practiced_at)
    conn.commit()


def _upsert_phoneme_state(
    conn: sqlite3.Connection,
    user_id: int,
    phoneme: str,
    alpha: float,
    beta: float,
    attempts_count: int,
    occurrence_count: int,
    last_practiced_at: str,
) -> None:
    """Upsert one phoneme state WITHOUT committing (transaction-friendly)."""
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


def get_sentence_by_text(text: str, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    conn = conn or get_connection()
    return conn.execute("SELECT * FROM exercise_bank WHERE text = ?", (text,)).fetchone()


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
    _complete_latest_assignment(conn, user_id, exercise_id, attempt_id)
    conn.commit()


def _complete_latest_assignment(
    conn: sqlite3.Connection, user_id: int, exercise_id: int, attempt_id: int
) -> None:
    """Stamp the latest open assignment complete WITHOUT committing."""
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
def _trusted_filter() -> str:
    """SQL fragment: only count attempts whose evidence actually updated
    mastery -- i.e. scorable audio scored by a TRUSTED (fully-validated
    PanPhon) engine. Legacy rows default mastery_updated=0, so they are
    treated as untrusted/unknown and excluded until explicitly migrated."""
    return "a.mastery_updated = 1"


def get_phoneme_context_stats(
    user_id: int, conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Dict[str, Any]]:
    """Per expected phoneme: how many independent recordings it appeared in
    and across how many distinct prompt texts. Used by evidence-aware level
    eligibility (>= N recordings across >= M prompts)."""
    conn = conn or get_connection()
    rows = conn.execute(
        f"""SELECT e.expected_phoneme AS phoneme,
                   a.id AS attempt_id,
                   a.text AS prompt_text,
                   COALESCE(a.quality_weight, 1.0) AS quality_weight
            FROM attempt_phoneme_events e
            JOIN attempts a ON e.attempt_id = a.id
            WHERE a.user_id = ?
              AND e.expected_phoneme IS NOT NULL
              AND {_trusted_filter()}""",
        (user_id,),
    ).fetchall()
    accumulated: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        state = accumulated.setdefault(
            row["phoneme"],
            {"attempt_ids": set(), "prompt_texts": set(), "attempt_weights": {}, "occurrences": 0},
        )
        state["attempt_ids"].add(row["attempt_id"])
        state["prompt_texts"].add(row["prompt_text"])
        state["attempt_weights"][row["attempt_id"]] = max(
            0.0, min(1.0, float(row["quality_weight"]))
        )
        state["occurrences"] += 1
    return {
        phoneme: {
            "recordings": len(state["attempt_ids"]),
            "effective_recordings": round(sum(state["attempt_weights"].values()), 6),
            "distinct_prompts": len(state["prompt_texts"]),
            "occurrences": state["occurrences"],
        }
        for phoneme, state in accumulated.items()
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
                  AND {_trusted_filter()}
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


def get_trusted_recording_count(user_id: int, conn: Optional[sqlite3.Connection] = None) -> int:
    """Number of the user's recordings that were scorable AND scored by a
    trusted engine (i.e. actually updated mastery)."""
    conn = conn or get_connection()
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM attempts a WHERE a.user_id = ? AND {_trusted_filter()}",
        (user_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


def get_effective_recording_count(user_id: int, conn: Optional[sqlite3.Connection] = None) -> float:
    """Quality-weighted trusted recording evidence for a saved profile."""
    conn = conn or get_connection()
    row = conn.execute(
        f"""SELECT COALESCE(SUM(
                    CASE
                        WHEN a.quality_weight IS NULL THEN 1.0
                        WHEN a.quality_weight < 0 THEN 0.0
                        WHEN a.quality_weight > 1 THEN 1.0
                        ELSE a.quality_weight
                    END
                ), 0.0) AS n
            FROM attempts a
            WHERE a.user_id = ? AND {_trusted_filter()}""",
        (user_id,),
    ).fetchone()
    return float(row["n"]) if row else 0.0


def get_utterance_epenthesis_state(
    user_id: int, conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """Build a quality-weighted Beta state for insertion-free utterances.

    The observation for one recording is ``max(0, 1 - insertions/ref_units)``.
    It lives at utterance level; no inserted phoneme is assigned to a reference
    phoneme or written into per-phoneme mastery.
    """
    conn = conn or get_connection()
    rows = conn.execute(
        f"""SELECT a.id, a.insertion_count, a.reference_unit_count,
                   COALESCE(a.quality_weight, 1.0) AS quality_weight,
                   (SELECT COUNT(*) FROM attempt_phoneme_events e
                    WHERE e.attempt_id = a.id AND e.expected_phoneme IS NOT NULL) AS event_ref_units
            FROM attempts a
            WHERE a.user_id = ? AND {_trusted_filter()}""",
        (user_id,),
    ).fetchall()

    alpha = beta = 1.0
    effective_evidence = 0.0
    total_insertions = 0
    included = 0
    for row in rows:
        reference_units = int(row["reference_unit_count"] or row["event_ref_units"] or 0)
        if reference_units <= 0:
            continue
        insertions = max(0, int(row["insertion_count"] or 0))
        weight = max(0.0, min(1.0, float(row["quality_weight"])))
        observation = max(0.0, 1.0 - (insertions / reference_units))
        alpha += weight * observation
        beta += weight * (1.0 - observation)
        effective_evidence += weight
        total_insertions += insertions
        included += 1

    return {
        "alpha": alpha,
        "beta": beta,
        "recordings": included,
        "effective_recordings": round(effective_evidence, 6),
        "insertion_count": total_insertions,
    }
