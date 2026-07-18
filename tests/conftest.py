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
