"""
IPA normalization and phoneme tokenization.

The one correctness rule here: a non-boundary chunk of IPA is ALWAYS split
into phonemes through ``split_ipa_word`` and every produced phoneme is
canonicalized. That makes tokenization behave identically for both input
shapes the app has to handle:

    Formatted reference IPA:  "s k uː l | ɪ z | oʊ p ə n"
    Raw decoded CTC output:   "skuːl ɪz oʊpən"

both ->  ["s", "k", "u", "l", "ɪ", "z", "oʊ", "p", "ə", "n"]

The previous implementation short-circuited on whitespace and returned whole
words ("skuːl", "ɪz", "oʊpən") for the CTC case, which then never aligned
against the per-phoneme reference.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List

from phoneme_vectors_professional import COMPOSITE_COMPONENTS, canonicalize_phoneme

# Multi-codepoint phonemes that must be recognized as ONE unit inside a word
# (affricates and diphthongs). Length symbols like /ː/ are handled separately
# as combining marks below. Sorted longest-first at match time.
MULTI_CHAR_PHONEMES: List[str] = list(COMPOSITE_COMPONENTS.keys())

# Marks that attach to the preceding phoneme rather than forming their own.
COMBINING_MARKS = {"ː", "̃", "̩", "̯", "ʰ", "˞"}
CTC_CONTROL_TOKENS = ("[PAD]", "[UNK]", "<pad>", "<unk>", "<s>", "</s>")


def normalize_ipa(text: str) -> str:
    """Canonicalize spacing and strip stress/bracket noise from an IPA string.
    Word boundaries are preserved as a spaced ``|`` token."""
    text = unicodedata.normalize("NFC", str(text))
    for control_token in CTC_CONTROL_TOKENS:
        text = text.replace(control_token, "")
    text = text.replace("ˈ", "").replace("ˌ", "")
    text = text.replace(":", "ː")
    text = text.replace("/", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("|", " | ")
    return " ".join(text.split())


def split_ipa_word(ipa_word: str) -> List[str]:
    """Split one IPA chunk into phoneme-like tokens.

    Example: ``skuːl`` -> ``["s", "k", "uː", "l"]`` (``uː`` canonicalizes to
    ``u`` downstream). Greedily matches multi-character phonemes first, then
    attaches combining marks (length, nasalization, ...) to the phoneme they
    modify.
    """
    ipa_word = normalize_ipa(ipa_word).replace(" ", "")
    if not ipa_word:
        return []

    tokens: List[str] = []
    i = 0
    ordered = sorted(MULTI_CHAR_PHONEMES, key=len, reverse=True)
    while i < len(ipa_word):
        matched = None
        for ph in ordered:
            if ipa_word.startswith(ph, i):
                matched = ph
                break
        if matched is not None:
            tokens.append(matched)
            i += len(matched)
            continue

        char = ipa_word[i]
        if char in COMBINING_MARKS and tokens:
            tokens[-1] += char
        elif char not in {" ", "_", "|"}:
            tokens.append(char)
        i += 1

    return tokens


def _tokenize(ipa: str) -> List[str]:
    """Core tokenizer shared by both public entry points: split every
    non-boundary chunk through ``split_ipa_word`` and canonicalize."""
    ipa = normalize_ipa(ipa)
    tokens: List[str] = []
    for chunk in ipa.split():
        if chunk == "|":
            continue
        tokens.extend(split_ipa_word(chunk))
    return [canon for canon in (canonicalize_phoneme(tok) for tok in tokens) if canon]


def tokenize_reference_ipa(ipa: str) -> List[str]:
    """Tokenize formatted reference IPA ("s k uː l | ɪ z") into canonical
    phonemes."""
    return _tokenize(ipa)


def tokenize_ctc_prediction(ipa: str) -> List[str]:
    """Tokenize raw decoded CTC output ("skuːl ɪz oʊpən") into canonical
    phonemes."""
    return _tokenize(ipa)


def ipa_to_tokens(ipa: str) -> List[str]:
    """Backward-compatible entry point. Works for BOTH the formatted-reference
    and raw-CTC shapes because both go through ``split_ipa_word``."""
    return _tokenize(ipa)


def words_to_spaced_ipa(ipa_words: List[str]) -> str:
    """Join per-word G2P outputs as phoneme-spaced words separated by ``|``."""
    spaced_words = []
    for word_ipa in ipa_words:
        tokens = split_ipa_word(word_ipa)
        if tokens:
            spaced_words.append(" ".join(tokens))
    return " | ".join(spaced_words)


# -----------------------------------------------------------------------------
# IPA reading guide (UI helper)
# -----------------------------------------------------------------------------
PHONEME_GUIDE: Dict[str, Dict[str, str]] = {
    "i": {"example": "see", "description": "long ee sound, like 'see'"},
    "ɪ": {"example": "sit", "description": "short i sound, like 'sit'"},
    "eɪ": {"example": "say", "description": "like 'ay' in 'say'"},
    "ɛ": {"example": "bed", "description": "short e sound, like 'bed'"},
    "æ": {"example": "cat", "description": "a sound like 'cat'"},
    "ɑ": {"example": "father", "description": "open ah sound, like 'father'"},
    "ɔ": {"example": "thought", "description": "aw sound, like 'thought'"},
    "ʊ": {"example": "foot", "description": "short oo sound, like 'foot'"},
    "u": {"example": "food", "description": "oo sound, like 'food'"},
    "ʌ": {"example": "cup", "description": "uh sound, like 'cup'"},
    "ə": {"example": "about", "description": "schwa: weak 'uh' sound"},
    "ɝ": {"example": "bird", "description": "r-colored vowel, like 'bird'"},
    "aɪ": {"example": "my", "description": "like 'eye', as in 'my'"},
    "aʊ": {"example": "now", "description": "like 'ow', as in 'now'"},
    "oʊ": {"example": "go", "description": "like 'oh', as in 'go'"},
    "ɔɪ": {"example": "boy", "description": "like 'oy', as in 'boy'"},
    "p": {"example": "pen", "description": "voiceless p sound"},
    "b": {"example": "boy", "description": "voiced b sound"},
    "t": {"example": "top", "description": "voiceless t sound"},
    "d": {"example": "dog", "description": "voiced d sound"},
    "k": {"example": "cat", "description": "voiceless k sound"},
    "ɡ": {"example": "go", "description": "voiced g sound"},
    "f": {"example": "fish", "description": "voiceless f sound"},
    "v": {"example": "van", "description": "voiced v sound"},
    "θ": {"example": "think", "description": "voiceless th sound, like 'think'"},
    "ð": {"example": "this", "description": "voiced th sound, like 'this'"},
    "s": {"example": "see", "description": "s sound"},
    "z": {"example": "zoo", "description": "z sound"},
    "ʃ": {"example": "she", "description": "sh sound"},
    "ʒ": {"example": "measure", "description": "zh sound, like 'measure'"},
    "h": {"example": "hat", "description": "h sound"},
    "tʃ": {"example": "chair", "description": "ch sound"},
    "dʒ": {"example": "jump", "description": "j sound"},
    "m": {"example": "man", "description": "m sound"},
    "n": {"example": "no", "description": "n sound"},
    "ŋ": {"example": "sing", "description": "ng sound, like the end of 'sing'"},
    "l": {"example": "love", "description": "l sound"},
    "ɹ": {"example": "red", "description": "English r sound"},
    "w": {"example": "we", "description": "w sound"},
    "j": {"example": "yes", "description": "y sound, like 'yes'"},
}


def ipa_reading_guide(ipa: str):
    """Word-by-word reading guide for an IPA sequence (UI display)."""
    ipa = normalize_ipa(ipa)
    words = [w.strip() for w in ipa.split("|") if w.strip()]
    guide = []
    for word_index, word in enumerate(words, start=1):
        phonemes = []
        for token in split_ipa_word(word):
            token = canonicalize_phoneme(token)
            info = PHONEME_GUIDE.get(token, {
                "example": "unknown",
                "description": "No guide available for this phoneme yet.",
            })
            phonemes.append({
                "symbol": token,
                "example": info["example"],
                "description": info["description"],
            })
        guide.append({"word_index": word_index, "phonemes": phonemes})
    return guide
