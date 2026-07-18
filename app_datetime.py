"""
Small shared datetime helpers for persisting/reading mastery timestamps.

Extracted so services.py and app.py agree on the exact format and both keep
``PhonemeStat.last_practiced_at`` tz-aware (UTC). Naive and aware datetimes
cannot be subtracted, and the decay math depends on that subtraction, so every
timestamp that reaches mastery.py must be aware.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parse_db_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored timestamp back into an aware UTC datetime."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def format_db_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")
