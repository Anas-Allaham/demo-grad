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

from phoneme_vectors_professional import canonicalize_phoneme, validate_g2p_inventory

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

# Minimum times a target phoneme should appear in an accepted exercise so the
# learner actually gets repetitions of it.
MIN_TARGET_REPETITIONS = int(os.environ.get("EXERCISE_MIN_TARGET_REPS", "2"))

# Candidate-scoring weights (documented, not magic numbers).
TARGET_OCCURRENCE_CAP = 3          # cap per-phoneme reps so we don't reward tongue-twisters
W_TARGET_OCCURRENCE = 1.0          # weight on capped target-phoneme occurrences
W_TARGET_COVERAGE = 1.5            # weight on number of distinct targets covered
W_UNDER_OBSERVED = 0.6            # weight on covering under-observed phonemes
W_DIVERSITY = 0.4                 # weight on word/context diversity
W_CONFUSION = 1.2                # bonus for covering the confused-with phoneme
W_DIFFICULTY_FIT = 2.0            # penalty weight for difficulty mismatch
W_OVERMASTERED = 0.05            # gentle: mastered sounds are useful scaffolding
W_LENGTH = 0.04                  # gentle length preference
W_EXCESS_LENGTH = 0.5            # extra penalty once past MAX_WORD_COUNT

# Difficulty scale. level_proxy ~ word_count + avg_word_len/2; normalize it to
# [0, 1]. This is a provisional readability proxy, NOT a validated CEFR band.
LEVEL_PROXY_MIN = 6.0
LEVEL_PROXY_MAX = 26.0

# Target difficulty per assessed level (provisional).
LEVEL_DIFFICULTY = {
    "unknown": 0.4,
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.8,
}

_client = None


# Word-count bands per exercise type (drives real selection, not just a label).
SHORT_PHRASE_MAX_WORDS = 5
TARGETED_SENTENCE_MAX_WORDS = 14
MAINTENANCE_MIN_WORDS = 10

# Minimal pairs for the most common English confusions. Keyed as a frozenset so
# lookup is order-independent (θ/s == s/θ). Each entry lists (word_a, word_b)
# pairs that isolate exactly that contrast -- the core material for low-mastery
# / confusion practice.
MINIMAL_PAIRS: Dict[frozenset, List[tuple]] = {
    frozenset({"θ", "s"}): [("think", "sink"), ("thick", "sick"), ("thing", "sing"), ("mouth", "mouse")],
    frozenset({"ð", "d"}): [("they", "day"), ("though", "dough"), ("breathe", "breed")],
    frozenset({"v", "f"}): [("van", "fan"), ("vine", "fine"), ("leave", "leaf")],
    frozenset({"w", "v"}): [("wine", "vine"), ("west", "vest"), ("worse", "verse")],
    frozenset({"ɪ", "i"}): [("ship", "sheep"), ("bit", "beat"), ("sit", "seat"), ("fill", "feel")],
    frozenset({"ʊ", "u"}): [("full", "fool"), ("pull", "pool")],
    frozenset({"l", "ɹ"}): [("light", "right"), ("lace", "race"), ("glass", "grass")],
    frozenset({"n", "ŋ"}): [("thin", "thing"), ("sin", "sing"), ("ban", "bang")],
    frozenset({"z", "s"}): [("zip", "sip"), ("zoo", "sue"), ("prize", "price")],
    frozenset({"b", "p"}): [("bat", "pat"), ("bin", "pin"), ("cab", "cap")],
    frozenset({"d", "t"}): [("dime", "time"), ("den", "ten"), ("bad", "bat")],
    frozenset({"ɛ", "æ"}): [("bed", "bad"), ("pen", "pan"), ("said", "sad")],
    frozenset({"ʃ", "s"}): [("ship", "sip"), ("shy", "sigh"), ("shell", "sell")],
    frozenset({"tʃ", "ʃ"}): [("chip", "ship"), ("chew", "shoe"), ("watch", "wash")],
}

# A few isolated words per phoneme for low-mastery practice when no confusion
# pair applies.
ISOLATED_WORDS: Dict[str, List[str]] = {
    "θ": ["think", "three", "bath"], "ð": ["this", "mother", "breathe"],
    "s": ["see", "grass", "bus"], "z": ["zoo", "buzz", "easy"],
    "ʃ": ["she", "wash", "ocean"], "ʒ": ["measure", "vision"],
    "tʃ": ["chair", "watch", "teacher"], "dʒ": ["jump", "bridge", "giant"],
    "v": ["van", "love", "seven"], "f": ["fish", "coffee", "leaf"],
    "ɹ": ["red", "very", "car"], "l": ["light", "yellow", "ball"],
    "w": ["water", "away"], "j": ["yes", "yellow"],
    "ŋ": ["sing", "long", "finger"], "ɪ": ["sit", "ship", "him"],
    "i": ["see", "green", "meet"], "ʊ": ["book", "put"], "u": ["food", "blue"],
    "ɛ": ["bed", "red"], "æ": ["cat", "bad"], "ɑ": ["father", "hot"],
    "ɔ": ["thought", "ball"], "ʌ": ["cup", "sun"], "ə": ["about", "sofa"],
    "ɝ": ["bird", "her", "world"], "aɪ": ["my", "time"], "aʊ": ["now", "house"],
    "eɪ": ["day", "cake"], "oʊ": ["go", "home"], "ɔɪ": ["boy", "toy"],
    "p": ["pen", "apple"], "b": ["boy", "table"], "t": ["top", "water"],
    "d": ["dog", "red"], "k": ["cat", "book"], "ɡ": ["go", "big"],
    "h": ["hat", "behind"], "m": ["man", "swim"], "n": ["no", "sun"],
}


def minimal_pair_words(phoneme_a: str, phoneme_b: str) -> Optional[List[tuple]]:
    """Return minimal-pair word tuples that contrast the two phonemes, if any."""
    return MINIMAL_PAIRS.get(frozenset({canonicalize_phoneme(phoneme_a), canonicalize_phoneme(phoneme_b)}))


def build_word_exercise(
    words: List[str],
    g2p_convert: Callable[[str], str],
    ipa_to_tokens: Callable[[str], List[str]],
    source: str,
) -> Optional[Dict]:
    """Tag a short list of words (an isolated-word or minimal-pair drill) with
    the SAME G2P pipeline used for scoring, so it is a first-class, verified
    exercise -- not a special-cased string."""
    text = " ".join(words)
    tagged = tag_sentence(text, g2p_convert, ipa_to_tokens)
    if not is_valid_tagging(tagged):
        return None
    tagged["source"] = source
    return tagged


def make_low_mastery_exercise(
    target: str,
    confused_with: Optional[str],
    g2p_convert: Callable[[str], str],
    ipa_to_tokens: Callable[[str], List[str]],
) -> Optional[Dict]:
    """Isolated-words / minimal-pairs drill for a low-mastery target. Prefers a
    minimal pair against the learner's actual confusion, else isolated words."""
    if confused_with:
        pairs = minimal_pair_words(target, confused_with)
        if pairs:
            words: List[str] = []
            for a, b in pairs[:3]:
                words.extend([a, b])
            ex = build_word_exercise(words, g2p_convert, ipa_to_tokens, source="minimal_pairs")
            if ex is not None:
                return ex
    words = ISOLATED_WORDS.get(canonicalize_phoneme(target), [])
    if words:
        return build_word_exercise(words[:3], g2p_convert, ipa_to_tokens, source="isolated_words")
    return None


def normalize_difficulty(level_proxy: float) -> float:
    """Map a raw level_proxy onto a [0, 1] difficulty scale (provisional)."""
    span = max(LEVEL_PROXY_MAX - LEVEL_PROXY_MIN, 1e-6)
    return max(0.0, min(1.0, (float(level_proxy) - LEVEL_PROXY_MIN) / span))


def difficulty_for_level(overall_level: Optional[str]) -> float:
    """Target difficulty for a learner level (provisional)."""
    return LEVEL_DIFFICULTY.get(overall_level or "unknown", 0.4)


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
    level_proxy: Optional[float] = None,
    target_difficulty: Optional[float] = None,
    under_observed_phonemes: Optional[Iterable[str]] = None,
    confusion_phonemes: Optional[Iterable[str]] = None,
) -> float:
    """Score a candidate sentence for how well it fits the learner right now.

    Considers (all documented above as weighted terms):
      * target-phoneme OCCURRENCE counts, capped so a sentence isn't rewarded
        for cramming one sound unnaturally;
      * how many DISTINCT target phonemes it covers;
      * CONTRASTIVE value: covering the phoneme the learner confuses the target
        with (so retrieval, not just LLM generation, is confusion-aware);
      * coverage of under-observed phonemes (diagnostic value);
      * word/context diversity;
      * fit between the sentence difficulty (from level_proxy) and the
        learner's level;
      * a GENTLE length preference and an excess-length penalty.

    Mastered phonemes are only gently penalized -- natural sentences need them
    and they provide useful scaffolding.
    """
    target_phonemes = list(target_phonemes)
    overmastered_phonemes = set(overmastered_phonemes)
    under_observed_phonemes = set(under_observed_phonemes or ())
    confusion_phonemes = set(confusion_phonemes or ())

    # Target occurrences (capped) + distinct-target coverage.
    occurrence_score = sum(
        min(phoneme_counts.get(p, 0), TARGET_OCCURRENCE_CAP) for p in target_phonemes
    )
    coverage = sum(1 for p in target_phonemes if p in phoneme_counts)

    # Contrastive bonus: a sentence covering BOTH the target and the sound it is
    # confused with lets the learner practise the distinction.
    confusion_cover = sum(1 for p in confusion_phonemes if p in phoneme_counts)

    under_observed_cover = sum(1 for p in under_observed_phonemes if p in phoneme_counts)

    # Distinct phonemes as a cheap word/context-diversity proxy.
    diversity = len(phoneme_counts)

    # Difficulty fit: penalize distance between sentence difficulty and target.
    difficulty_penalty = 0.0
    if level_proxy is not None and target_difficulty is not None:
        difficulty_penalty = abs(normalize_difficulty(level_proxy) - target_difficulty)

    overmastered_penalty = sum(1 for p in phoneme_counts if p in overmastered_phonemes)
    length_penalty = abs(word_count - target_word_count)
    excess_length = max(0, word_count - MAX_WORD_COUNT)

    return (
        W_TARGET_OCCURRENCE * occurrence_score
        + W_TARGET_COVERAGE * coverage
        + W_CONFUSION * confusion_cover
        + W_UNDER_OBSERVED * under_observed_cover
        + W_DIVERSITY * math.log1p(diversity)
        - W_DIFFICULTY_FIT * difficulty_penalty
        - W_OVERMASTERED * overmastered_penalty
        - W_LENGTH * length_penalty
        - W_EXCESS_LENGTH * excess_length
    )


def pick_next_sentence(
    candidates: List[Dict],
    target_phonemes: List[str],
    overmastered_phonemes: Optional[Iterable[str]] = None,
    recently_served_ids: Optional[Iterable[int]] = None,
    target_word_count: int = TARGET_WORD_COUNT,
    target_difficulty: Optional[float] = None,
    under_observed_phonemes: Optional[Iterable[str]] = None,
    confusion_phonemes: Optional[Iterable[str]] = None,
    word_count_range: Optional[tuple] = None,
) -> Optional[Dict]:
    """Best-scoring candidate covering at least one target phoneme, biased away
    from recently served sentences and toward a difficulty that fits the
    learner. ``word_count_range`` (min, max) restricts to a length band for the
    exercise type (short phrase / sentence / connected speech); it relaxes
    rather than returning nothing. Falls back to a recent sentence if needed."""
    overmastered_phonemes = set(overmastered_phonemes or ())
    recently_served_ids = set(recently_served_ids or ())
    under_observed_phonemes = set(under_observed_phonemes or ())
    confusion_phonemes = set(confusion_phonemes or ())

    if not candidates:
        return None

    pool = [c for c in candidates if c["id"] not in recently_served_ids] or candidates
    if word_count_range is not None:
        lo, hi = word_count_range
        banded = [c for c in pool if lo <= c["word_count"] <= hi]
        pool = banded or pool  # relax the band rather than serve nothing

    return max(
        pool,
        key=lambda c: score_candidate(
            c["phoneme_counts"],
            target_phonemes,
            overmastered_phonemes,
            c["word_count"],
            target_word_count,
            level_proxy=c.get("level_proxy"),
            target_difficulty=target_difficulty,
            under_observed_phonemes=under_observed_phonemes,
            confusion_phonemes=confusion_phonemes,
        ),
    )


def pick_diagnostic_sentence(
    all_sentences: List[Dict],
    recently_served_ids: Optional[Iterable[int]] = None,
    target_word_count: int = TARGET_WORD_COUNT,
    uncovered_phonemes: Optional[Iterable[str]] = None,
) -> Optional[Dict]:
    """Diagnostic choice.

    Without ``uncovered_phonemes`` this is the classic cold-start pick: the
    broadest-coverage sentence not served recently. With ``uncovered_phonemes``
    (the phonemes still lacking enough independent observations) it instead
    maximizes NEW coverage -- the number of still-uncovered phonemes the
    sentence would exercise -- so the diagnostic phase keeps probing sounds the
    first sentence missed instead of re-drilling what it already saw.
    """
    recently_served_ids = set(recently_served_ids or ())
    if not all_sentences:
        return None
    eligible = [s for s in all_sentences if s["id"] not in recently_served_ids] or all_sentences

    uncovered = set(uncovered_phonemes or ())
    if uncovered:
        return max(
            eligible,
            key=lambda s: (
                sum(1 for p in uncovered if p in s["phoneme_counts"]),   # new-coverage gain
                len(s["phoneme_counts"]),
                -abs(s["word_count"] - target_word_count),
            ),
        )
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


_LEVEL_GUIDANCE = {
    "beginner": "Use short, common, everyday words and simple grammar (CEFR A1-A2 style).",
    "intermediate": "Use moderately varied vocabulary and natural sentence structure (CEFR B1-B2 style).",
    "advanced": "You may use richer vocabulary and more complex structure (CEFR C1 style).",
    "unknown": "Use clear, natural, everyday language of moderate difficulty.",
}


def generate_candidate_text(
    target_phonemes: List[str],
    avoid_phonemes: Iterable[str],
    level: Optional[str] = None,
    confusion_hint: Optional[str] = None,
) -> Optional[str]:
    """One LLM-proposed sentence. Returns None on any failure (no key, no
    package, API error) so the caller falls back to the retrieval bank --
    generation is a bonus, never a hard dependency.

    The learner ``level`` is now actually included in the prompt so difficulty
    is steered, and a ``confusion_hint`` (e.g. "θ vs s") requests contrastive
    material for a known confusion.
    """
    client = _get_client()
    if client is None:
        return None

    avoid_list = ", ".join(sorted(avoid_phonemes)) or "none"
    level = (level or "unknown").lower()
    level_line = _LEVEL_GUIDANCE.get(level, _LEVEL_GUIDANCE["unknown"])
    confusion_line = (
        f"The learner tends to confuse {confusion_hint}; include contrasting words that "
        "make that distinction clear.\n"
        if confusion_hint else ""
    )
    prompt = (
        f"Write ONE natural English sentence of at most {MAX_WORD_COUNT} words for a "
        "pronunciation-practice app.\n"
        f"Learner level: {level}. {level_line}\n"
        f"Focus tightly on these {len(target_phonemes)} IPA target sound(s): "
        f"{', '.join(target_phonemes)}.\n"
        f"Include each target sound at least {MIN_TARGET_REPETITIONS} times using natural words, so the "
        "learner gets repetitions -- but keep the sentence grammatical and meaningful, not a "
        "random word list.\n"
        f"{confusion_line}"
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


def meets_min_repetitions(
    phoneme_counts: Dict[str, int],
    target_phonemes: List[str],
    min_reps: int = MIN_TARGET_REPETITIONS,
) -> bool:
    """At least one target phoneme must appear ``min_reps`` times so the
    exercise gives real repetition of a target sound."""
    if not target_phonemes:
        return True
    return any(phoneme_counts.get(p, 0) >= min_reps for p in target_phonemes)


def verify_generated_exercise(
    tagged: Dict,
    target_phonemes: List[str],
    overmastered_phonemes: Iterable[str],
    min_reps: int = MIN_TARGET_REPETITIONS,
) -> tuple[bool, List[str]]:
    """Check a generated (already-tagged) exercise against acceptance rules.
    Returns (ok, reasons_for_rejection). Rules: valid/supported G2P output,
    max length, target coverage, minimum target repetitions, and not padded
    with already-mastered sounds."""
    reasons: List[str] = []
    if not is_valid_tagging(tagged):
        reasons.append("untaggable_or_oov")
    else:
        report = validate_g2p_inventory(tagged["phoneme_counts"].keys())
        if not report["ok"]:
            reasons.append(f"unsupported_phonemes:{report['unsupported']}")
    if tagged.get("word_count", 0) > MAX_WORD_COUNT:
        reasons.append("too_long")
    if not covers_targets(tagged.get("phoneme_counts", {}), target_phonemes):
        reasons.append("insufficient_target_coverage")
    if not meets_min_repetitions(tagged.get("phoneme_counts", {}), target_phonemes, min_reps):
        reasons.append("insufficient_target_repetitions")
    if too_many_overmastered(tagged.get("phoneme_counts", {}), overmastered_phonemes):
        reasons.append("overmastered_padding")
    return (not reasons, reasons)


def generate_and_verify_exercise(
    target_phonemes: List[str],
    overmastered_phonemes: Iterable[str],
    g2p_convert: Callable[[str], str],
    ipa_to_tokens: Callable[[str], List[str]],
    level: Optional[str] = None,
    confusion_hint: Optional[str] = None,
    max_attempts: int = GENERATION_MAX_ATTEMPTS,
) -> Optional[Dict]:
    """Rejection-sampling loop: ask the LLM for a candidate, verify it with the
    SAME G2P pipeline used for scoring (never a separate/simplified check),
    accept only a candidate that passes every rule in
    ``verify_generated_exercise``. Returns None -- never a silently-unverified
    sentence -- if nothing passes within ``max_attempts``.

    ``level`` is the learner's assessed level (steers difficulty in the
    prompt); ``confusion_hint`` requests contrastive material.
    """
    overmastered_phonemes = set(overmastered_phonemes)
    for _ in range(max_attempts):
        candidate_text = generate_candidate_text(
            target_phonemes, overmastered_phonemes, level=level, confusion_hint=confusion_hint
        )
        if not candidate_text:
            return None  # no client available or the call failed -- don't keep retrying
        tagged = tag_sentence(candidate_text, g2p_convert, ipa_to_tokens)
        ok, _reasons = verify_generated_exercise(tagged, target_phonemes, overmastered_phonemes)
        if ok:
            tagged["source"] = "llm_generated"
            return tagged
    return None
