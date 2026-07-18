"""Audit #6 & #7: functional exercise types + confusion retrieval; insertions."""

import content
from g2p_service import g2p_convert
from tokenization import ipa_to_tokens
from scoring import align_phonemes, calculate_metrics


# ---- #6: minimal pairs / isolated words for low mastery ----
def test_minimal_pair_lookup_is_order_independent():
    assert content.minimal_pair_words("θ", "s") == content.minimal_pair_words("s", "θ")
    assert content.minimal_pair_words("θ", "s")  # non-empty for a known confusion


def test_low_mastery_builds_minimal_pair_drill():
    ex = content.make_low_mastery_exercise("θ", "s", g2p_convert, ipa_to_tokens)
    assert ex is not None
    assert ex["source"] == "minimal_pairs"
    # It really contrasts θ and s and is short (isolated words, not a sentence).
    assert "think" in ex["text"] and "sink" in ex["text"]
    assert ex["word_count"] <= 8


def test_low_mastery_falls_back_to_isolated_words():
    ex = content.make_low_mastery_exercise("ŋ", None, g2p_convert, ipa_to_tokens)
    assert ex is not None
    assert ex["source"] == "isolated_words"


# ---- #6: exercise_type drives length band ----
def test_word_count_band_selection():
    from services import _word_count_range_for
    assert _word_count_range_for("short_phrase") == (1, content.SHORT_PHRASE_MAX_WORDS)
    assert _word_count_range_for("maintenance")[0] == content.MAINTENANCE_MIN_WORDS

    short = {"id": 1, "text": "a", "phoneme_counts": {"s": 2}, "word_count": 3, "level_proxy": 5}
    long = {"id": 2, "text": "b", "phoneme_counts": {"s": 2}, "word_count": 18, "level_proxy": 20}
    picked = content.pick_next_sentence([short, long], ["s"], word_count_range=(1, content.SHORT_PHRASE_MAX_WORDS))
    assert picked["id"] == 1  # the short one, within band


# ---- #6: retrieval is confusion-aware (not only LLM) ----
def test_retrieval_prefers_contrastive_sentence():
    plain = {"id": 1, "text": "a", "phoneme_counts": {"θ": 2, "k": 1}, "word_count": 6, "level_proxy": 8}
    contrastive = {"id": 2, "text": "b", "phoneme_counts": {"θ": 2, "s": 2}, "word_count": 6, "level_proxy": 8}
    picked = content.pick_next_sentence(
        [plain, contrastive], ["θ"], confusion_phonemes={"s"}
    )
    assert picked["id"] == 2  # covers the confused-with sound too


# ---- #7: insertions influence utterance score but not per-phoneme mastery ----
def test_insertions_tracked_at_utterance_level_only():
    # Reference "s", hypothesis "s s s" -> 2 insertions, no expected phoneme.
    rows = align_phonemes(["s"], ["s", "s", "s"])
    m = calculate_metrics(rows)
    assert m["insertion_count"] == 2
    assert m["insertion_penalty"] > 0
    # Utterance score is degraded by the insertions...
    assert m["utterance_score"] < 100.0
    # ...but no insertion row carries an expected phoneme (so mastery can't see it).
    for r in rows:
        if r["result"] == "insertion":
            assert r["expected"] == "-"


def test_insertions_do_not_update_phoneme_mastery():
    import mastery
    from datetime import datetime, timezone
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = align_phonemes(["s"], ["s", "t", "t"])  # extra t's are insertions
    stats = mastery.update_mastery_for_recording({}, rows, now)
    # Only the expected phoneme 's' gets evidence; inserted 't' does not appear.
    assert set(stats.keys()) == {"s"}
