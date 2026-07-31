"""Convert internal IPA data into the Flask/browser ARPAbet contract."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from phoneme_alphabet import (
    ARPABET_TO_IPA,
    IPAInputFormat,
    canonicalize_arpabet_phoneme,
    ipa_to_arpabet,
    ipa_word_to_arpabet_tokens,
)


_SEQUENCE_FORMATS = {
    "ipa": IPAInputFormat.FORMATTED_REFERENCE,
    "reference_ipa": IPAInputFormat.FORMATTED_REFERENCE,
    "predicted_ipa": IPAInputFormat.RAW_CTC,
}
_RENAMED_KEYS = {
    "ipa": "arpabet",
    "reference_ipa": "reference_arpabet",
    "predicted_ipa": "predicted_arpabet",
}
_SINGLE_PHONEME_KEYS = frozenset({"phoneme", "symbol", "expected", "spoken"})
_PHONEME_LIST_KEYS = frozenset({
    "target_phonemes",
    "unknown_phonemes",
    "strong_phonemes",
    "uncovered",
    "panphon_inventory_failures",
    "canonical_inventory_unsupported",
})


def ipa_phoneme_to_arpabet(
    phoneme: Any,
    *,
    strict: bool = True,
    allow_ctc_fallbacks: bool = True,
) -> Any:
    if phoneme is None or phoneme == "-":
        return phoneme
    tokens = ipa_word_to_arpabet_tokens(
        str(phoneme),
        strict=strict,
        allow_ctc_fallbacks=allow_ctc_fallbacks,
    )
    return " ".join(tokens)


def _convert_confusion_hint(value: str) -> str:
    if " vs " not in value:
        return value
    expected, spoken = value.split(" vs ", 1)
    return f"{ipa_phoneme_to_arpabet(expected)} vs {ipa_phoneme_to_arpabet(spoken)}"


def to_public_arpabet(value: Any, parent_key: str | None = None) -> Any:
    """Return a recursively converted copy for JSON/UI output."""
    if isinstance(value, Mapping):
        converted: Dict[str, Any] = {}
        for key, child in value.items():
            converted[_RENAMED_KEYS.get(key, key)] = to_public_arpabet(child, key)
        return converted

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if parent_key in _PHONEME_LIST_KEYS:
            return [ipa_phoneme_to_arpabet(item) for item in value]
        return [to_public_arpabet(item, parent_key) for item in value]

    if isinstance(value, str):
        if parent_key in _SEQUENCE_FORMATS:
            input_format = _SEQUENCE_FORMATS[parent_key]
            return ipa_to_arpabet(
                value,
                input_format=input_format,
                strict=input_format is IPAInputFormat.FORMATTED_REFERENCE,
            )
        if parent_key in _SINGLE_PHONEME_KEYS:
            return ipa_phoneme_to_arpabet(
                value,
                strict=parent_key != "spoken",
            )
        if parent_key == "confusion_hint":
            return _convert_confusion_hint(value)

    return value


def metrics_to_internal_ipa(metrics: Mapping[str, float] | None) -> Dict[str, float]:
    """Convert public ARPAbet-keyed metrics to the internal IPA inventory."""
    converted: Dict[str, float] = {}
    for raw_phoneme, score in (metrics or {}).items():
        arpabet = canonicalize_arpabet_phoneme(str(raw_phoneme))
        ipa = ARPABET_TO_IPA.get(arpabet)
        if ipa is not None:
            converted[ipa] = float(score)
    return converted
