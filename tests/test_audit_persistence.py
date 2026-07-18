"""Audit #8 & #9 (+ migration): temp audio, atomic writes, non-destructive migration."""

import sqlite3
from pathlib import Path

import pytest


# ---- #9: attempt + events + mastery + assignment are one atomic transaction ----
def test_atomic_recording_rolls_back_on_failure(temp_db, monkeypatch):
    db = temp_db
    user = db.get_or_create_user("atomic")
    uid = user["id"]

    alignment = [{"expected": "s", "spoken": "s", "result": "correct",
                  "articulatory_distance": 0.0, "alignment_cost": 0.0}]

    # Force a failure AFTER the attempt + events are inserted but before commit.
    def boom(*a, **k):
        raise RuntimeError("simulated mid-transaction failure")

    monkeypatch.setattr(db, "_complete_latest_assignment", boom)

    with pytest.raises(RuntimeError):
        db.record_recording_atomic(
            user_id=uid,
            attempt_kwargs={"text": "t", "reference_ipa": "s", "predicted_ipa": "s",
                            "phoneme_error_rate": 0.0, "weighted_error": 0.0, "scorable": True,
                            "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True},
            alignment=alignment, scoring_engine="panphon",
            phoneme_states={"s": {"alpha": 2.0, "beta": 1.0, "attempts_count": 1,
                                  "occurrence_count": 1, "last_practiced_at": "2026-01-01 00:00:00"}},
            complete_exercise_id=1,
        )

    conn = db.get_connection()
    # Nothing partially written: no attempt, no events, no mastery state.
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM attempt_phoneme_events").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM phoneme_skill_state").fetchone()["n"] == 0


def test_atomic_recording_commits_all_on_success(temp_db):
    db = temp_db
    user = db.get_or_create_user("ok")
    uid = user["id"]
    aid = db.record_recording_atomic(
        user_id=uid,
        attempt_kwargs={"text": "t", "reference_ipa": "s", "predicted_ipa": "s",
                        "phoneme_error_rate": 0.0, "weighted_error": 0.0, "scorable": True,
                        "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True},
        alignment=[{"expected": "s", "spoken": "s", "result": "correct",
                    "articulatory_distance": 0.0, "alignment_cost": 0.0}],
        scoring_engine="panphon",
        phoneme_states={"s": {"alpha": 2.0, "beta": 1.0, "attempts_count": 1,
                              "occurrence_count": 1, "last_practiced_at": "2026-01-01 00:00:00"}},
    )
    conn = db.get_connection()
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM attempt_phoneme_events").fetchone()["n"] == 1
    assert conn.execute("SELECT alpha FROM phoneme_skill_state WHERE phoneme='s'").fetchone()["alpha"] == 2.0


# ---- #8: temporary audio deletion ----
def test_cleanup_deletes_audio_by_default(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "RETAIN_AUDIO", False)
    files = []
    for name in ("orig.webm", "orig_converted.wav", "orig_reduced.wav"):
        p = tmp_path / name
        p.write_bytes(b"\x00\x01")
        files.append(p)
    app._cleanup_audio_files(files)
    assert all(not p.exists() for p in files)


def test_cleanup_keeps_audio_when_retention_enabled(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "RETAIN_AUDIO", True)
    p = tmp_path / "keep.wav"
    p.write_bytes(b"\x00")
    app._cleanup_audio_files([p])
    assert p.exists()


# ---- Non-destructive additive migration ----
def test_migration_is_additive_and_preserves_data(tmp_path):
    import db

    old_db = tmp_path / "old.db"
    conn = sqlite3.connect(str(old_db))
    conn.row_factory = sqlite3.Row
    # Minimal OLD schema (pre-audit): no trust columns, legacy phoneme keys.
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, created_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE exercise_bank (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT UNIQUE, reference_ipa TEXT, word_count INTEGER, level_proxy REAL DEFAULT 0, source TEXT DEFAULT 'retrieval', created_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE sentence_phonemes (sentence_id INTEGER, phoneme TEXT, count INTEGER DEFAULT 1, PRIMARY KEY (sentence_id, phoneme));
        CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, exercise_id INTEGER, text TEXT, reference_ipa TEXT, predicted_ipa TEXT, phoneme_error_rate REAL, weighted_error REAL, created_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE attempt_phoneme_events (id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id INTEGER, position INTEGER, expected_phoneme TEXT, spoken_phoneme TEXT, operation TEXT, distance REAL);
        CREATE TABLE phoneme_skill_state (user_id INTEGER, phoneme TEXT, alpha REAL DEFAULT 1, beta REAL DEFAULT 1, attempts_count INTEGER DEFAULT 0, last_practiced_at TEXT, PRIMARY KEY (user_id, phoneme));
        CREATE TABLE practice_assignments (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, exercise_id INTEGER, target_phonemes TEXT, assigned_at TEXT DEFAULT (datetime('now')), completed_attempt_id INTEGER);
        INSERT INTO users (id, name) VALUES (1, 'legacy');
        INSERT INTO attempts (user_id, text, reference_ipa, predicted_ipa, phoneme_error_rate, weighted_error) VALUES (1, 'hi', 'h aɪ', 'h aɪ', 5.0, 0.2);
        -- legacy non-canonical keys that must merge on migration:
        INSERT INTO phoneme_skill_state (user_id, phoneme, alpha, beta, attempts_count) VALUES (1, 'r', 3.0, 1.0, 2);
        INSERT INTO phoneme_skill_state (user_id, phoneme, alpha, beta, attempts_count) VALUES (1, 'ɹ', 2.0, 1.0, 1);
        INSERT INTO phoneme_skill_state (user_id, phoneme, alpha, beta, attempts_count) VALUES (1, 'iː', 4.0, 1.0, 3);
        """
    )
    conn.commit()
    conn.close()

    db.set_database_for_testing(old_db)
    try:
        db.init_db()  # runs the additive migration + canonicalization
        conn = db.get_connection()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(attempts)").fetchall()}
        assert {"scoring_engine", "scoring_trusted", "mastery_updated", "insertion_count"} <= cols

        # User data preserved.
        assert conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"] == 1

        # Legacy keys canonicalized: r+ɹ merged to ɹ, iː -> i. No non-canonical keys.
        from phoneme_vectors_professional import canonicalize_phoneme
        keys = [r["phoneme"] for r in conn.execute("SELECT phoneme FROM phoneme_skill_state").fetchall()]
        assert all(canonicalize_phoneme(k) == k for k in keys)
        assert "ɹ" in keys and "i" in keys and "r" not in keys and "iː" not in keys
    finally:
        db.set_database_for_testing(Path(__file__).resolve().parent.parent / "app.db")
