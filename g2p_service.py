"""
Trusted grapheme-to-phoneme (G2P) service.

Heteronym resolution lives in this lightweight module and is applied before
either the optional NeMo backend or the bundled dictionary backend. This is
important: installing NeMo can improve non-dictionary coverage, but it is not
what turns contextual heteronym handling on.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tokenization import normalize_ipa, tokenize_reference_ipa, words_to_spaced_ipa

BASE_DIR = Path(__file__).resolve().parent
G2P_DIR = BASE_DIR / "g2p_pipeline_split_v2"
HETERONYMS_PATH = G2P_DIR / "heteronyms.json"
IPA_DICT_PATH = G2P_DIR / "cmudict-0.7b-ipa.txt"

g2p_engine = None
g2p_mode = "not_loaded"


@dataclass(frozen=True)
class ReferenceG2PResult:
    text: str
    ipa: str
    g2p_mode: str
    heteronym_resolution_active: bool
    reference_g2p_trusted: bool
    unresolved_heteronyms: Tuple[str, ...] = ()
    unsupported_heteronyms: Tuple[str, ...] = ()
    oov_words: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "g2p_mode": self.g2p_mode,
            "heteronym_resolution_active": self.heteronym_resolution_active,
            "reference_g2p_trusted": self.reference_g2p_trusted,
            "unresolved_heteronyms": list(self.unresolved_heteronyms),
            "unsupported_heteronyms": list(self.unsupported_heteronyms),
            "oov_words": list(self.oov_words),
        }


class UntrustedReferenceG2PError(ValueError):
    """Raised when trusted exercise generation receives an unsafe reference."""


def _candidate_to_ipa(candidate: Any) -> str:
    if isinstance(candidate, list):
        return "".join(str(part) for part in candidate)
    return str(candidate or "")


class ContextualHeteronymResolver:
    """Resolve the bundled heteronym lexicon without importing NeMo.

    A local spaCy tagger is used when its English model is available. A small,
    deterministic context tagger covers explicit noun/verb/adjective cues when
    it is not. Ambiguous contexts use the lexicon default for display but are
    marked untrusted, so they cannot become mastery evidence.
    """

    _DETERMINERS = {
        "a", "an", "the", "this", "that", "these", "those", "my", "your",
        "his", "her", "its", "our", "their", "each", "every", "no",
    }
    _SUBJECTS = {"i", "you", "we", "they", "he", "she", "it"}
    _VERB_CUES = {
        "please", "to", "can", "could", "may", "might", "must", "shall",
        "should", "will", "would", "do", "does", "did", "don't", "not",
    }
    _ADJECTIVE_CUES = {"be", "become", "feel", "get", "keep", "look", "remain", "seem", "stay", "very", "too"}
    _PAST_CUES = {"yesterday", "ago", "last", "earlier", "previously"}
    _PRESENT_CUES = {"daily", "usually", "often", "always", "today", "every"}

    def __init__(self, json_path: str | Path = HETERONYMS_PATH) -> None:
        self.json_path = Path(json_path)
        with self.json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.entries: Dict[str, Dict[str, Any]] = {
            str(word).lower(): entry for word, entry in data.items()
        }
        self._spacy_nlp = self._load_optional_spacy()
        self.unsupported_contrasts = set(
            validate_heteronym_lexicon(self.json_path)["unsupported_contrasts"]
        )

    @staticmethod
    def _load_optional_spacy():
        try:
            import spacy

            return spacy.load("en_core_web_sm")
        except Exception:
            return None

    def contains(self, word: str) -> bool:
        return word.lower() in self.entries

    def _spacy_tags(self, text: str) -> List[str]:
        if self._spacy_nlp is None:
            return []
        doc = self._spacy_nlp(text)
        return [token.tag_ for token in doc if re.search(r"[A-Za-z']", token.text)]

    def _heuristic_tag(self, words: Sequence[str], index: int) -> Optional[str]:
        word = words[index]
        previous = words[index - 1] if index else ""
        following = words[index + 1] if index + 1 < len(words) else ""
        context = set(words)

        if word == "read":
            if context & self._PAST_CUES or previous in {"had", "has", "have"}:
                return "VBD" if previous not in {"had", "has", "have"} else "VBN"
            if context & self._PRESENT_CUES or ("every" in context and "day" in context):
                return "VBP"
            return None

        if previous in self._ADJECTIVE_CUES:
            return "JJ"
        if previous in self._VERB_CUES or previous in self._SUBJECTS:
            return "VB" if previous in self._VERB_CUES else "VBP"
        if previous in self._DETERMINERS:
            # A determiner followed by a copula/preposition is a strong noun cue.
            if following in {"is", "was", "are", "were", "has", "of", "for"}:
                return "NN"
            return "NN"
        return None

    def resolve(
        self,
        word: str,
        words: Sequence[str],
        index: int,
        spacy_tags: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        entry = self.entries.get(word.lower())
        if entry is None:
            return None

        pos_map = entry.get("pos_map", {})
        tag = spacy_tags[index] if index < len(spacy_tags) else None
        method = "spacy" if tag in pos_map else "heuristic"
        if tag not in pos_map:
            tag = self._heuristic_tag(words, index)

        unsupported = word.lower() in self.unsupported_contrasts
        if tag in pos_map:
            return {
                "ipa": _candidate_to_ipa(pos_map[tag]),
                "tag": tag,
                "method": method,
                "trusted": not unsupported,
                "unsupported": unsupported,
            }

        return {
            "ipa": _candidate_to_ipa(entry.get("default")),
            "tag": None,
            "method": "unresolved_default",
            "trusted": False,
            "unsupported": unsupported,
        }

    def tags_for_text(self, text: str) -> List[str]:
        return self._spacy_tags(text)


def validate_heteronym_lexicon(path: str | Path = HETERONYMS_PATH) -> Dict[str, Any]:
    """Validate schema, candidate tokenization, inventory, and contrast loss."""
    from phoneme_vectors_professional import in_inventory

    lexicon_path = Path(path)
    with lexicon_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    invalid_entries: List[str] = []
    unsupported_phonemes: Dict[str, List[str]] = {}
    unsupported_contrasts: List[str] = []
    for word, entry in data.items():
        pos_map = entry.get("pos_map") if isinstance(entry, dict) else None
        default = entry.get("default") if isinstance(entry, dict) else None
        if not isinstance(pos_map, dict) or not pos_map or default is None:
            invalid_entries.append(word)
            continue

        raw_pronunciations = set()
        canonical_pronunciations = set()
        candidates = list(pos_map.values()) + [default]
        for candidate in candidates:
            ipa = _candidate_to_ipa(candidate)
            tokens = tokenize_reference_ipa(ipa)
            if not ipa or not tokens:
                invalid_entries.append(word)
                continue
            raw_pronunciations.add(ipa)
            canonical_pronunciations.add(tuple(tokens))
            bad = sorted({token for token in tokens if not in_inventory(token)})
            if bad:
                unsupported_phonemes.setdefault(word, []).extend(bad)

        if len(canonical_pronunciations) < len(raw_pronunciations):
            unsupported_contrasts.append(word)

    invalid_entries = sorted(set(invalid_entries))
    unsupported_phonemes = {
        word: sorted(set(phonemes)) for word, phonemes in unsupported_phonemes.items()
    }
    checked = len(data)
    valid_entries = checked - len(invalid_entries)
    return {
        "checked": checked,
        "valid_entries": valid_entries,
        "invalid_entries": invalid_entries,
        "unsupported_phonemes": unsupported_phonemes,
        "unsupported_contrasts": sorted(set(unsupported_contrasts)),
        "schema_and_inventory_ok": not invalid_entries and not unsupported_phonemes,
        "fully_supported": not invalid_entries and not unsupported_phonemes and not unsupported_contrasts,
    }


def load_ipa_dictionary() -> Dict[str, str]:
    """Load the first bundled dictionary pronunciation for each word."""
    if not IPA_DICT_PATH.exists():
        raise FileNotFoundError(f"IPA dictionary not found: {IPA_DICT_PATH}")

    dictionary: Dict[str, str] = {}
    with IPA_DICT_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
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
            dictionary.setdefault(word, ipa.split(",")[0].strip())
    return dictionary


class DictionaryIpaG2p:
    """Context-aware bundled-dictionary G2P with explicit trust metadata."""

    mode = "context_aware_dictionary_fallback"
    heteronym_resolution_active = True

    def __init__(self, resolver: Optional[ContextualHeteronymResolver] = None) -> None:
        self.dictionary = load_ipa_dictionary()
        self.resolver = resolver or ContextualHeteronymResolver()

    def _lookup_non_heteronym(self, word: str) -> Tuple[str, bool]:
        ipa = self.dictionary.get(word)
        return (ipa, True) if ipa is not None else (word, False)

    def resolve(self, text: str) -> ReferenceG2PResult:
        words = [word.strip("'").lower() for word in re.findall(r"[A-Za-z']+", text)]
        spacy_tags = self.resolver.tags_for_text(text)
        ipa_words: List[str] = []
        unresolved: List[str] = []
        unsupported: List[str] = []
        oov: List[str] = []

        for index, word in enumerate(words):
            decision = self.resolver.resolve(word, words, index, spacy_tags)
            if decision is not None:
                ipa_words.append(decision["ipa"])
                if decision["unsupported"]:
                    unsupported.append(word)
                elif not decision["trusted"]:
                    unresolved.append(word)
                continue

            ipa, trusted = self._lookup_non_heteronym(word)
            ipa_words.append(ipa)
            if not trusted:
                oov.append(word)

        ipa = normalize_ipa(words_to_spaced_ipa(ipa_words))
        trusted = bool(ipa) and not unresolved and not unsupported and not oov
        return ReferenceG2PResult(
            text=text,
            ipa=ipa,
            g2p_mode=self.mode,
            heteronym_resolution_active=self.heteronym_resolution_active,
            reference_g2p_trusted=trusted,
            unresolved_heteronyms=tuple(sorted(set(unresolved))),
            unsupported_heteronyms=tuple(sorted(set(unsupported))),
            oov_words=tuple(sorted(set(oov))),
        )

    def __call__(self, text: str) -> List[str]:
        result = self.resolve(text)
        return [part.strip().replace(" ", "") for part in result.ipa.split("|") if part.strip()]


class NemoDictionaryIpaG2p(DictionaryIpaG2p):
    """The same resolver with NeMo used only for non-heteronym fallback."""

    mode = "context_aware_nemo_ipa_g2p"

    def __init__(self, nemo_g2p, resolver: Optional[ContextualHeteronymResolver] = None) -> None:
        super().__init__(resolver=resolver)
        self.nemo_g2p = nemo_g2p

    def _lookup_non_heteronym(self, word: str) -> Tuple[str, bool]:
        output = self.nemo_g2p(word)
        if isinstance(output, str):
            items = [output]
        elif isinstance(output, list):
            items = output
        else:
            items = [str(output)]
        cleaned = []
        for item in items:
            if item == ",":
                break
            cleaned.append(str(item))
        ipa = "".join(cleaned).strip()
        trusted = bool(ipa) and ipa.lower() != word.lower()
        return (ipa or word, trusted)


def load_g2p_engine() -> None:
    """Load heteronym resolution first, then select a non-heteronym backend."""
    global g2p_engine, g2p_mode
    if g2p_engine is not None:
        return

    if str(G2P_DIR) not in sys.path:
        sys.path.insert(0, str(G2P_DIR))

    resolver = ContextualHeteronymResolver()
    try:
        from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p

        backend = IpaG2p(
            phoneme_dict=str(IPA_DICT_PATH),
            locale="en-US",
            ignore_ambiguous_words=False,
            use_chars=False,
            use_stresses=True,
        )
        g2p_engine = NemoDictionaryIpaG2p(backend, resolver=resolver)
    except Exception as exc:
        print("NeMo IpaG2p could not be loaded. Using context-aware dictionary fallback.")
        print("Reason:", repr(exc))
        g2p_engine = DictionaryIpaG2p(resolver=resolver)
    g2p_mode = g2p_engine.mode


def g2p_convert_with_metadata(text: str) -> ReferenceG2PResult:
    load_g2p_engine()
    return g2p_engine.resolve(text)


def g2p_convert(text: str) -> str:
    """Backward-compatible text -> spaced IPA conversion."""
    return g2p_convert_with_metadata(text).ipa


def g2p_convert_trusted(text: str) -> str:
    """Text -> IPA, rejecting references unsafe for exercise verification."""
    result = g2p_convert_with_metadata(text)
    if not result.reference_g2p_trusted:
        details = result.unresolved_heteronyms + result.unsupported_heteronyms + result.oov_words
        raise UntrustedReferenceG2PError(
            "Reference G2P is unresolved or unsupported: " + ", ".join(details)
        )
    return result.ipa


def get_g2p_mode() -> str:
    load_g2p_engine()
    return g2p_mode


def heteronym_resolution_active() -> bool:
    load_g2p_engine()
    return bool(getattr(g2p_engine, "heteronym_resolution_active", False))
