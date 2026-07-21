"""Saved assessments, quality evidence, intervals, and epenthesis agree."""

from datetime import datetime, timezone

import app as flask_app
import assessment as A
import mastery

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
PHONEMES = ["s", "z", "t", "d", "n", "m", "k", "p", "f", "l", "ɹ", "i"]


def _profile(alpha=5.0, beta=4.0, effective=4.0):
    stats = {
        phoneme: mastery.PhonemeStat(
            alpha=alpha,
            beta=beta,
            independent_attempts=4,
            occurrence_count=6,
            last_practiced_at=NOW,
        )
        for phoneme in PHONEMES
    }
    context = {
        phoneme: {
            "recordings": 4,
            "effective_recordings": effective,
            "distinct_prompts": 2,
            "occurrences": 6,
        }
        for phoneme in PHONEMES
    }
    return stats, context


def test_quality_weighted_evidence_is_required_for_level_eligibility():
    stats, context = _profile(alpha=8.0, beta=2.0, effective=0.8)
    result = A.assess_user_level(
        stats,
        context,
        independent_recording_count=4,
        effective_recording_count=0.8,
        now=NOW,
    )
    assert result["eligible_phoneme_count"] == 0
    assert result["assessment_status"] == "insufficient_evidence"
    assert result["overall_level"] == "unknown"


def test_interval_crossing_threshold_reports_uncertain_borderline():
    stats, context = _profile(alpha=5.0, beta=4.0)
    result = A.assess_user_level(stats, context, 4, now=NOW)
    low, high = result["credible_interval"]
    assert low < A.BEGINNER_MAX_SCORE <= high
    assert result["overall_level"] == "uncertain"
    assert result["level_decision"] == "borderline"
    assert result["borderline_levels"] == ["beginner", "intermediate"]
    assert result["exercise_level"] == "beginner"


def test_utterance_epenthesis_state_changes_overall_profile_without_phoneme_attachment():
    stats, context = _profile(alpha=8.0, beta=2.0)
    clean = A.assess_user_level(
        stats,
        context,
        10,
        now=NOW,
        effective_recording_count=10.0,
        utterance_state={"alpha": 11.0, "beta": 1.0, "effective_recordings": 10.0, "insertion_count": 0},
    )
    epenthetic = A.assess_user_level(
        stats,
        context,
        10,
        now=NOW,
        effective_recording_count=10.0,
        utterance_state={"alpha": 1.0, "beta": 11.0, "effective_recordings": 10.0, "insertion_count": 30},
    )
    assert epenthetic["pronunciation_score"] < clean["pronunciation_score"]
    assert epenthetic["utterance_epenthesis_state"]["insertion_count"] == 30
    assert epenthetic["utterance_epenthesis_state"]["profile_weight"] == 0.2
    assert set(stats) == set(PHONEMES)


def test_db_epenthesis_state_uses_quality_weight_and_no_reference_phoneme(temp_db):
    user = temp_db.get_or_create_user("epenthesis-state")
    temp_db.record_attempt(
        user_id=user["id"],
        text="test",
        reference_ipa="s t a r t",
        predicted_ipa="s s t a r t",
        phoneme_error_rate=20.0,
        weighted_error=0.85,
        raw_weighted_per=20.0,
        quality_weight=0.5,
        scorable=True,
        scoring_engine="panphon",
        scoring_trusted=True,
        mastery_updated=True,
        insertion_count=2,
        reference_unit_count=4,
        reference_g2p_trusted=True,
    )
    state = temp_db.get_utterance_epenthesis_state(user["id"])
    # Observation = 1 - 2/4 = 0.5, weighted by 0.5.
    assert state["alpha"] == 1.25
    assert state["beta"] == 1.25
    assert state["effective_recordings"] == 0.5
    assert temp_db.get_all_phoneme_states(user["id"]) == []


def test_saved_exercise_uses_profile_assessment_and_raw_metrics_stay_stateless(temp_db, monkeypatch):
    monkeypatch.setattr(flask_app, "RETAIN_AUDIO", False)
    sentence_id = temp_db.insert_sentence(
        text="School is open.",
        reference_ipa="s k u l | ɪ z | oʊ p ə n",
        word_count=3,
        level_proxy=5.0,
        phoneme_counts={"s": 2, "k": 1, "u": 1, "l": 1},
    )
    assert sentence_id is not None
    client = flask_app.app.test_client()

    saved = client.get("/exercise?user=assessment-consistency").get_json()
    assert saved["assessment"]["assessment_source"] == "saved_profile_evidence"
    assert saved["assessment"]["raw_stateless_metrics"] is False

    raw = client.post("/exercise", json={"metrics": {}}).get_json()
    assert raw["assessment"]["assessment_source"] == "stateless_raw_metrics"
    assert raw["assessment"]["raw_stateless_metrics"] is True
    assert raw["assessment"]["credible_interval"] is None


def test_saved_exercise_stays_diagnostic_with_sparse_eligible_weakness(temp_db):
    import services
    from g2p_service import g2p_convert_with_metadata
    from tokenization import ipa_to_tokens

    user = temp_db.get_or_create_user("sparse-diagnostic")
    temp_db.upsert_phoneme_state(
        user_id=user["id"], phoneme="s", alpha=1.5, beta=5.5,
        attempts_count=4, occurrence_count=4, last_practiced_at="2026-01-01 00:00:00",
    )
    alignment = [{
        "expected": "s", "spoken": "z", "result": "close_substitution",
        "articulatory_distance": 0.2, "alignment_cost": 0.45,
    }]
    for index in range(4):
        temp_db.record_recording_atomic(
            user_id=user["id"],
            attempt_kwargs={
                "text": f"prompt-{index % 2}", "reference_ipa": "s", "predicted_ipa": "z",
                "phoneme_error_rate": 45.0, "weighted_error": 0.45,
                "quality_weight": 1.0, "scorable": True, "scoring_engine": "panphon",
                "scoring_trusted": True, "mastery_updated": True,
                "reference_unit_count": 1, "reference_g2p_trusted": True,
            },
            alignment=alignment,
            scoring_engine="panphon",
        )
    temp_db.insert_sentence(
        text="School is open.", reference_ipa="s k u l | ɪ z | oʊ p ə n",
        word_count=3, level_proxy=5.0, phoneme_counts={"s": 2, "k": 1, "u": 1},
    )

    result = services.generate_exercise_for_profile(
        user["id"], g2p_convert_with_metadata, ipa_to_tokens, now=NOW
    )
    assert result["assessment"]["eligible_phoneme_count"] == 1
    assert result["assessment"]["assessment_status"] == "insufficient_evidence"
    assert result["source_mode"] == "diagnostic"
    assert result["exercise_type"] == "diagnostic"
    assert result["target_phonemes"] == []


def test_stateless_metrics_drop_nonfinite_values_and_clamp_range():
    import services

    result = services.provisional_assessment_from_metrics({"s": float("nan"), "t": 2.0, "d": -1.0})
    assert result["eligible_phoneme_count"] == 2
    assert result["pronunciation_score"] == 50.0
    assert result["overall_level"] == "beginner"
    assert result["strong_phonemes"] == ["t"]
    assert [item["phoneme"] for item in result["weak_phonemes"]] == ["d"]
