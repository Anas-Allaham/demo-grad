"""
Practice-sentence selection: retrieval from the offline-tagged exercise
bank first, an optional LLM-generation-with-G2P-verification fallback
second.

Nothing here calls the database directly — callers (app.py, the offline
tagging script) pass in plain dicts/lists so this stays trivially testable
without spinning up sqlite or the audio pipeline. `tag_sentence` takes the
app's own `g2p_convert`/`ipa_to_tokens` functions as parameters rather than
importing app.py, which both avoids a circular import (app.py imports this
module) and guarantees tagging always uses the exact same G2P path the
sentence will later be scored against.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional

# Exercise generation talks to an LLM through an OpenAI-compatible endpoint,
# via the `openai` client library. Defaults target Google AI Studio (Gemini),
# but everything is env-configurable so the same code works against any
# OpenAI-compatible provider (Gemini, Qwen/DashScope, OpenAI, ...) with no
# code edits -- just change the env vars.
try:
    from openai import OpenAI
except Exception as exc:
    OpenAI = None
    print("openai package is unavailable. LLM exercise generation is disabled.")
    print("Reason:", repr(exc))

# Model. Default is Gemini's fast, free-tier model.
#   Gemini:  gemini-flash-latest  (default) | gemini-2.0-flash
#   Qwen:    qwen-plus | qwen-turbo | qwen-flash   (also set LLM_BASE_URL below)
LLM_MODEL = os.environ.get("EXERCISE_LLM_MODEL", "gemini-flash-latest")

# OpenAI-compatible base URL. Default is Google AI Studio (Gemini). For Qwen:
#   LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# API key, checked in order. GEMINI_API_KEY / GOOGLE_API_KEY are Gemini's
# standard names; the others let the same code pick up a Qwen key too.
LLM_API_KEY_ENV_VARS = (
    "LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY",
)

# Output-token budget per generation call. Gemini's flash models spend tokens
# on internal "thinking" before the answer, so a small budget can get eaten
# before any sentence is produced -- keep this generous (the sentence itself
# is tiny; this is headroom for the model's reasoning).
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "800"))

# Preferred sentence length for scoring/generation, and the hard upper bound.
# Sentences may run up to MAX_WORD_COUNT words so a generated exercise can
# pack in many repetitions of the target sounds; TARGET_WORD_COUNT is just
# the length the retrieval scorer gently prefers.
TARGET_WORD_COUNT = int(os.environ.get("EXERCISE_TARGET_WORDS", "12"))
MAX_WORD_COUNT = int(os.environ.get("EXERCISE_MAX_WORDS", "30"))
COVERAGE_MIN_FRACTION = 0.6
OVERMASTERED_MAX_FRACTION = 0.5
GENERATION_MAX_ATTEMPTS = 5

_client = None


def _has_oov_fallback_words(text: str, reference_ipa: str) -> bool:
    """Detect words the G2P engine couldn't actually convert.

    The dictionary-only G2P fallback (used when NeMo/spaCy aren't
    installed) doesn't fail loudly on an out-of-vocabulary word -- it
    returns the word's own spelling in place of its IPA (see
    `DictionaryIpaG2p` in app.py), which then tokenizes into letters
    rather than sounds (e.g. "sandcastle" -> s-a-n-d-c-a-s-t-l-e). Some of
    those letters coincidentally collide with real single-character IPA
    symbols (s, n, d, m, ...), so checking the tokenized *phonemes* can't
    reliably catch this. Checking at the *word* level does: `g2p_convert`
    returns phonemes grouped per word by "|", so if a word's IPA segment
    (spaces stripped) is identical to its own spelling, that word was
    never actually converted.
    """
    words = [w.strip("'").lower() for w in re.findall(r"[A-Za-z']+", text)]
    ipa_words = [segment.replace(" ", "") for segment in reference_ipa.split("|")]
    if len(words) != len(ipa_words):
        return False  # can't align word-for-word -- don't guess
    return any(word == ipa_word for word, ipa_word in zip(words, ipa_words))


def tag_sentence(
    text: str,
    g2p_convert: Callable[[str], str],
    ipa_to_tokens: Callable[[str], List[str]],
) -> Dict:
    """Run the app's own G2P pipeline over `text` and record what it
    contains: phoneme counts, word count, and a difficulty proxy.

    The difficulty proxy is deliberately simple (word count + average word
    length) rather than a validated CEFR classifier -- it's documented here
    as an approximation, not asserted as linguistically calibrated.
    """
    reference_ipa = g2p_convert(text)
    tokens = ipa_to_tokens(reference_ipa)
    phoneme_counts = dict(Counter(tokens))
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    level_proxy = round(word_count + avg_word_len / 2, 2)
    return {
        "text": text,
        "reference_ipa": reference_ipa,
        "phoneme_counts": phoneme_counts,
        "word_count": word_count,
        "level_proxy": level_proxy,
        "has_oov_words": _has_oov_fallback_words(text, reference_ipa),
    }


def is_valid_tagging(tagged: Dict) -> bool:
    """Reject sentences the G2P pipeline couldn't meaningfully tokenize:
    empty output, or at least one word that fell through to the
    out-of-vocabulary raw-spelling fallback (see `_has_oov_fallback_words`)."""
    if not tagged.get("phoneme_counts"):
        return False
    return not tagged.get("has_oov_words", False)


# -----------------------------
# Retrieval scoring/selection
# -----------------------------
def score_candidate(
    phoneme_counts: Dict[str, int],
    target_phonemes: Iterable[str],
    overmastered_phonemes: Iterable[str],
    word_count: int,
    target_word_count: int = TARGET_WORD_COUNT,
) -> float:
    overmastered_phonemes = set(overmastered_phonemes)
    overlap = sum(1 for p in target_phonemes if p in phoneme_counts)
    overmastered_penalty = sum(1 for p in phoneme_counts if p in overmastered_phonemes) * 0.15
    length_penalty = abs(word_count - target_word_count) * 0.05
    return overlap - overmastered_penalty - length_penalty


def pick_next_sentence(
    candidates: List[Dict],
    target_phonemes: List[str],
    overmastered_phonemes: Optional[Iterable[str]] = None,
    recently_served_ids: Optional[Iterable[int]] = None,
    target_word_count: int = TARGET_WORD_COUNT,
) -> Optional[Dict]:
    """Best-scoring candidate covering at least one target phoneme, biased
    away from ones served recently and from overloading already-mastered
    sounds. Falls back to repeating a recent sentence rather than serving
    nothing if every candidate was recently served."""
    overmastered_phonemes = set(overmastered_phonemes or ())
    recently_served_ids = set(recently_served_ids or ())

    if not candidates:
        return None

    eligible = [c for c in candidates if c["id"] not in recently_served_ids] or candidates
    return max(
        eligible,
        key=lambda c: score_candidate(
            c["phoneme_counts"], target_phonemes, overmastered_phonemes, c["word_count"], target_word_count
        ),
    )


def pick_diagnostic_sentence(
    all_sentences: List[Dict],
    recently_served_ids: Optional[Iterable[int]] = None,
    target_word_count: int = TARGET_WORD_COUNT,
) -> Optional[Dict]:
    """Cold-start choice: the broadest-coverage sentence (most distinct
    phonemes) not recently served, used before any mastery data exists."""
    recently_served_ids = set(recently_served_ids or ())
    if not all_sentences:
        return None
    eligible = [s for s in all_sentences if s["id"] not in recently_served_ids] or all_sentences
    return max(
        eligible,
        key=lambda s: (len(s["phoneme_counts"]), -abs(s["word_count"] - target_word_count)),
    )


# -----------------------------
# LLM-generation fallback, verified by the same G2P pipeline
# -----------------------------
def _get_api_key() -> Optional[str]:
    for var in LLM_API_KEY_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


def _get_client():
    global _client
    if OpenAI is None or not _get_api_key():
        return None
    if _client is None:
        try:
            _client = OpenAI(api_key=_get_api_key(), base_url=LLM_BASE_URL)
        except Exception as exc:
            print("Could not initialise the Qwen (OpenAI-compatible) client:", repr(exc))
            return None
    return _client


def llm_available() -> bool:
    return _get_client() is not None


def generate_candidate_text(
    target_phonemes: List[str],
    avoid_phonemes: Iterable[str],
    level_hint: float,
) -> Optional[str]:
    """One LLM-proposed sentence via Qwen. Returns None on any failure (no
    key, no package, API error) so the caller falls back to the retrieval
    bank -- generation is a bonus, never a hard dependency."""
    client = _get_client()
    if client is None:
        return None

    avoid_list = ", ".join(sorted(avoid_phonemes)) or "none"
    prompt = (
        f"Write ONE natural English sentence of at most {MAX_WORD_COUNT} words for a "
        "pronunciation-practice app.\n"
        f"Focus tightly on these {len(target_phonemes)} IPA target sound(s): "
        f"{', '.join(target_phonemes)}.\n"
        "Pack in as many words containing those target sounds as you naturally can, so the "
        "learner gets lots of repetitions of them -- but keep the sentence grammatical and "
        "meaningful, not a random word list.\n"
        f"Avoid overusing these already-mastered sounds: {avoid_list}.\n"
        "Respond with ONLY the sentence itself -- no quotes, no explanation, no preamble."
    )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        print("LLM exercise generation call failed:", repr(exc))
        return None

    if not response.choices:
        return None
    text = (response.choices[0].message.content or "").strip()
    # Keep only the first line -- some models add a trailing note despite the
    # "sentence only" instruction.
    text = text.splitlines()[0].strip() if text else ""
    return text.strip("\"'").strip() or None


def covers_targets(phoneme_counts: Dict[str, int], target_phonemes: List[str], min_fraction: float = COVERAGE_MIN_FRACTION) -> bool:
    if not target_phonemes:
        return True
    present = sum(1 for p in target_phonemes if p in phoneme_counts)
    required = max(1, math.ceil(len(target_phonemes) * min_fraction))
    return present >= required


def too_many_overmastered(phoneme_counts: Dict[str, int], overmastered_phonemes: Iterable[str], max_fraction: float = OVERMASTERED_MAX_FRACTION) -> bool:
    if not phoneme_counts:
        return False
    overmastered_phonemes = set(overmastered_phonemes)
    overmastered_present = sum(1 for p in phoneme_counts if p in overmastered_phonemes)
    return (overmastered_present / len(phoneme_counts)) > max_fraction


def generate_and_verify_exercise(
    target_phonemes: List[str],
    overmastered_phonemes: Iterable[str],
    level_hint: float,
    g2p_convert: Callable[[str], str],
    ipa_to_tokens: Callable[[str], List[str]],
    max_attempts: int = GENERATION_MAX_ATTEMPTS,
) -> Optional[Dict]:
    """Rejection-sampling loop: ask the LLM for a candidate, verify it with
    the SAME G2P pipeline used for scoring (never a separate/simplified
    check), accept only if it actually covers the target phonemes without
    padding itself with already-mastered ones. Returns None -- never a
    silently-unverified sentence -- if nothing passes within max_attempts."""
    overmastered_phonemes = set(overmastered_phonemes)
    for _ in range(max_attempts):
        candidate_text = generate_candidate_text(target_phonemes, overmastered_phonemes, level_hint)
        if not candidate_text:
            return None  # no client available or the call failed -- don't keep retrying
        tagged = tag_sentence(candidate_text, g2p_convert, ipa_to_tokens)
        if not is_valid_tagging(tagged):
            continue
        if tagged["word_count"] > MAX_WORD_COUNT:
            continue  # too long -- reject and let the model try again
        if covers_targets(tagged["phoneme_counts"], target_phonemes) and not too_many_overmastered(
            tagged["phoneme_counts"], overmastered_phonemes
        ):
            tagged["source"] = "llm_generated"
            return tagged
    return None
