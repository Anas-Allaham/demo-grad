"""Server-side Cleanvoice preprocessing for recorded speech.

The browser never sees the API key. The official SDK uploads the local WAV,
waits for the asynchronous Cleanvoice job, and downloads the enhanced WAV to
the requested temporary path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    from cleanvoice import Cleanvoice
except Exception:
    Cleanvoice = None


class CleanvoiceProcessingError(RuntimeError):
    """Raised when configured Cleanvoice preprocessing cannot complete."""


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cleanvoice_enabled() -> bool:
    """Enable automatically when a key is present unless explicitly disabled."""
    has_key = bool(os.environ.get("CLEANVOICE_API_KEY", "").strip())
    return _env_flag("CLEANVOICE_ENABLED", has_key)


def cleanvoice_configured() -> bool:
    return cleanvoice_enabled() and bool(os.environ.get("CLEANVOICE_API_KEY", "").strip())


def _cleanvoice_class():
    """Load the SDK lazily so a running dev server can see a new installation."""
    global Cleanvoice
    if Cleanvoice is None:
        try:
            from cleanvoice import Cleanvoice as cleanvoice_class
        except Exception:
            return None
        Cleanvoice = cleanvoice_class
    return Cleanvoice


def cleanvoice_sdk_available() -> bool:
    return _cleanvoice_class() is not None


def cleanvoice_strict() -> bool:
    """When strict, an API failure stops analysis instead of using local cleanup."""
    return _env_flag("CLEANVOICE_STRICT", False)


def enhance_recording(input_path: Path, output_path: Path) -> Path:
    """Denoise and normalize one WAV with Cleanvoice, preserving speech content.

    Filler, silence, breath, and stutter removal stay disabled because cutting
    content or timing would make pronunciation assessment less trustworthy.
    """
    api_key = os.environ.get("CLEANVOICE_API_KEY", "").strip()
    if not cleanvoice_enabled():
        raise CleanvoiceProcessingError("Cleanvoice preprocessing is disabled.")
    if not api_key:
        raise CleanvoiceProcessingError("Cleanvoice is enabled but CLEANVOICE_API_KEY is missing.")
    cleanvoice_class = _cleanvoice_class()
    if cleanvoice_class is None:
        raise CleanvoiceProcessingError(
            "Cleanvoice is configured but cleanvoice-sdk is not installed."
        )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        timeout = max(10, int(float(os.environ.get("CLEANVOICE_HTTP_TIMEOUT", "120"))))
    except ValueError:
        timeout = 120

    client: Optional[Any] = None
    try:
        client = cleanvoice_class(api_key=api_key, timeout=timeout)
        client.process(
            str(input_path),
            remove_noise=True,
            normalize=True,
            studio_sound=_env_flag("CLEANVOICE_STUDIO_SOUND", False),
            fillers=False,
            long_silences=False,
            mouth_sounds=False,
            breath=False,
            stutters=False,
            transcription=False,
            summarize=False,
            social_content=False,
            export_format="wav",
            output_path=str(output_path),
        )
    except Exception as exc:
        # SDK errors can include signed storage URLs. Keep those server-side.
        raise CleanvoiceProcessingError(
            "Cleanvoice could not enhance this recording. Please try again."
        ) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise CleanvoiceProcessingError("Cleanvoice returned no enhanced audio.")
    return output_path
