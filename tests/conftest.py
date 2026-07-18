"""
Make the project root importable so tests can ``import scoring`` etc. without
installing the package. Tests deliberately avoid importing app.py (Wav2Vec2 /
torch) -- the acoustic layer is mocked or bypassed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture
def temp_db(tmp_path):
    """Point db at a fresh temp SQLite file for the duration of a test, then
    restore the original path. Keeps tests from touching the real app.db."""
    import db

    original = db.DB_PATH
    db.set_database_for_testing(tmp_path / "test_app.db")
    db.init_db()
    try:
        yield db
    finally:
        db.set_database_for_testing(original)

