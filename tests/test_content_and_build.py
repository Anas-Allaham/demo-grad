"""Tests 14-16: exercise selection, difficulty fit, and the bank builder."""

import importlib.util
from pathlib import Path

import content

ROOT = Path(__file__).resolve().parent.parent


# 14. Exercise selection favors more target repetitions.
def test_selection_favors_more_target_repetitions():
    common_kwargs = dict(
        target_phonemes=["θ"], overmastered_phonemes=set(), word_count=8,
        level_proxy=10.0, target_difficulty=0.4,
    )
    more = content.score_candidate(phoneme_counts={"θ": 3, "s": 2}, **common_kwargs)
    less = content.score_candidate(phoneme_counts={"θ": 1, "s": 2}, **common_kwargs)
    assert more > less


def test_target_occurrences_are_capped():
    # Beyond the cap, extra repetitions do not keep increasing the score
    # (avoids rewarding unnatural tongue-twisters).
    kwargs = dict(target_phonemes=["θ"], overmastered_phonemes=set(), word_count=8,
                  level_proxy=10.0, target_difficulty=0.4)
    at_cap = content.score_candidate(phoneme_counts={"θ": content.TARGET_OCCURRENCE_CAP}, **kwargs)
    over_cap = content.score_candidate(phoneme_counts={"θ": content.TARGET_OCCURRENCE_CAP + 5}, **kwargs)
    assert over_cap == at_cap


# 15. Exercise difficulty matches the assessed level.
def test_difficulty_matches_level():
    beginner_d = content.difficulty_for_level("beginner")
    advanced_d = content.difficulty_for_level("advanced")
    assert beginner_d < advanced_d

    # Same phoneme content and length; only sentence difficulty differs.
    def score(level_proxy, target_difficulty):
        return content.score_candidate(
            phoneme_counts={"θ": 2, "s": 1}, target_phonemes=["θ"],
            overmastered_phonemes=set(), word_count=10,
            level_proxy=level_proxy, target_difficulty=target_difficulty,
        )

    easy_proxy, hard_proxy = 7.0, 24.0
    # Beginner prefers the easy sentence; advanced prefers the hard one.
    assert score(easy_proxy, beginner_d) > score(hard_proxy, beginner_d)
    assert score(hard_proxy, advanced_d) > score(easy_proxy, advanced_d)


def test_recent_exercise_is_avoided_when_an_alternative_exists():
    recent = {"id": 1, "phoneme_counts": {"s": 3}, "word_count": 8, "level_proxy": 10.0}
    fresh = {"id": 2, "phoneme_counts": {"s": 2}, "word_count": 8, "level_proxy": 10.0}
    picked = content.pick_next_sentence([recent, fresh], ["s"], recently_served_ids={1})
    assert picked["id"] == 2


# 16. The exercise-bank builder locates the seed file.
def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_exercise_bank", ROOT / "scripts" / "build_exercise_bank.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_locates_seed_file():
    builder = _load_builder()
    assert builder.SEED_PATH.name == "seed_sentences.txt"
    assert builder.SEED_PATH.exists()
    sentences = builder.load_seed_sentences()
    assert len(sentences) > 0
    assert all(isinstance(s, str) and s for s in sentences)


def test_exercise_tagging_rejects_untrusted_reference_g2p():
    from g2p_service import g2p_convert_with_metadata
    from tokenization import ipa_to_tokens

    tagged = content.tag_sentence("They permit entry", g2p_convert_with_metadata, ipa_to_tokens)
    assert tagged["reference_g2p_trusted"] is False
    assert content.is_valid_tagging(tagged) is False
