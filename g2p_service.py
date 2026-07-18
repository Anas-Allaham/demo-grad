"""
Grapheme-to-phoneme (G2P) service.

Extracted from app.py so the offline exercise-bank builder and tests can run
the exact same text->IPA path WITHOUT importing app.py (and therefore without
pulling in torch / transformers / the Wav2Vec2 model). app.py and
scripts/build_exercise_bank.py both call ``g2p_convert`` from here, so a
sentence is always tagged with the same phonemes it will later be scored
against.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

from tokenization import normalize_ipa, words_to_spaced_ipa

BASE_DIR = Path(__file__).resolve().parent
G2P_DIR = BASE_DIR / "g2p_pipeline_split_v2"
HETERONYMS_PATH = G2P_DIR / "heteronyms.json"
IPA_DICT_PATH = G2P_DIR / "cmudict-0.7b-ipa.txt"

g2p_engine = None
g2p_mode = "not_loaded"


def load_ipa_dictionary() -> Dict[str, str]:
    """Local IPA dictionary fallback using cmudict-0.7b-ipa.txt."""
    if not IPA_DICT_PATH.exists():
        raise FileNotFoundError(f"IPA dictionary not found: {IPA_DICT_PATH}")

    dictionary: Dict[str, str] = {}
    with IPA_DICT_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            if "\t" in line:
                word, ipa = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                word, ipa = parts
            word = re.sub(r"\(\d+\)$", "", word).lower()
            ipa = ipa.split(",")[0].strip()
            dictionary.setdefault(word, ipa)
    return dictionary


class DictionaryIpaG2p:
    """Fallback G2P if NeMo or spaCy is not installed."""

    def __init__(self) -> None:
        self.dictionary = load_ipa_dictionary()

    def __call__(self, text: str) -> List[str]:
        words = re.findall(r"[A-Za-z']+", text.lower())
        out = []
        for word in words:
            lookup = word.strip("'")
            out.append(self.dictionary.get(lookup, lookup))
        return out


def load_g2p_engine() -> None:
    """Prefer the ContextAwareIpaG2p engine; fall back to the bundled IPA
    dictionary when NeMo/spaCy is unavailable."""
    global g2p_engine, g2p_mode
    if g2p_engine is not None:
        return

    if str(G2P_DIR) not in sys.path:
        sys.path.insert(0, str(G2P_DIR))

    try:
        from contextual_g2p import ContextAwareIpaG2p

        g2p_engine = ContextAwareIpaG2p(
            heteronyms_json_path=str(HETERONYMS_PATH),
            ipa_dict_path=str(IPA_DICT_PATH),
        )
        g2p_mode = "context_aware_ipa_g2p"
    except Exception as exc:
        print("ContextAwareIpaG2p could not be loaded. Using dictionary fallback.")
        print("Reason:", repr(exc))
        g2p_engine = DictionaryIpaG2p()
        g2p_mode = "dictionary_ipa_fallback"


def g2p_convert(text: str) -> str:
    """Text -> IPA phonemes, phonemes space-separated and words separated by
    ``|``."""
    load_g2p_engine()
    ipa_words = g2p_engine(text)
    return normalize_ipa(words_to_spaced_ipa(ipa_words))


def get_g2p_mode() -> str:
    return g2p_mode
