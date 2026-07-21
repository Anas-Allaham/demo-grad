"""Audit #10: end-to-end Flask route tests (acoustic/model layer mocked)."""

import io
from pathlib import Path

import pytest

import app as flask_app
from audio_quality import AudioQualityDecision


@pytest.fixture
def client(temp_db, monkeypatch):
    monkeypatch.setattr(flask_app, "RETAIN_AUDIO", False)
    # Seed a minimal exercise bank so retrieval-based routes have content.
    import content
    from g2p_service import g2p_convert
    from tokenization import ipa_to_tokens
    for text in ("The sun is bright today.", "She sells shells.",
                 "My brother reads good books.", "We go to school in the morning."):
        tagged = content.tag_sentence(text, g2p_convert, ipa_to_tokens)
        if content.is_valid_tagging(tagged):
            temp_db.insert_sentence(
                text=tagged["text"], reference_ipa=tagged["reference_ipa"],
                word_count=tagged["word_count"], level_proxy=tagged["level_proxy"],
                phoneme_counts=tagged["phoneme_counts"], source="retrieval",
            )
    return flask_app.app.test_client()


def _scorable_result(audio_path):
    return {
        "quality_decision": AudioQualityDecision(True, 0.95, [], {"envelope_modulation": 0.5}),
        "predicted_ipa": "s k u l",
        "reduced_audio_path": None,
        "noise_reduction_applied": False,
        "preprocessing_pipeline": "test",
        "cleanup_paths": [audio_path],
    }


def _unscorable_result(audio_path):
    return {
        "quality_decision": AudioQualityDecision(False, 0.0, ["no_speech_modulation"], {}),
        "predicted_ipa": None,
        "reduced_audio_path": None,
        "noise_reduction_applied": False,
        "preprocessing_pipeline": "not_processed",
        "cleanup_paths": [audio_path],
    }


def test_health_reports_readiness_and_trust(client):
    h = client.get("/health").get_json()
    assert h["status"] == "running"
    assert "scoring_trusted" in h and "panphon_inventory_ok" in h
    assert h["scoring_trusted"] == h["panphon_inventory_ok"]
    assert h["audio_retention_enabled"] is False
    assert h["heteronym_resolution_active"] is True
    assert h["heteronym_entries_checked"] == 72


def test_g2p_route(client):
    data = client.post("/g2p", json={"text": "school"}).get_json()
    assert data["ipa"]
    assert "s" in data["ipa"]
    assert data["heteronym_resolution_active"] is True
    assert data["reference_g2p_trusted"] is True


def test_practice_next_cold_start_is_diagnostic(client):
    r = client.get("/practice/next?user=__routes_new__").get_json()
    assert r["mode"] == "diagnostic"
    assert r["diagnostic"]["in_diagnostic"] is True


def test_analyze_scorable_updates_trusted_mastery_and_deletes_audio(client, monkeypatch):
    monkeypatch.setattr(flask_app, "process_recording", _scorable_result)
    uploads = flask_app.UPLOAD_FOLDER
    before = set(uploads.glob("*"))

    data = {
        "text": "school",
        "user": "__routes_user__",
        "audio": (io.BytesIO(b"RIFFfake"), "rec.webm", "audio/webm"),
    }
    resp = client.post("/analyze", data=data, content_type="multipart/form-data").get_json()

    assert resp["scorable"] is True
    assert resp["scoring_trusted"] is True          # PanPhon is installed + validated
    assert resp["mastery_updated"] is True
    assert "utterance_score" in resp["metrics"]
    # #8: no audio artifact left behind.
    after = set(uploads.glob("*"))
    assert after == before


def test_analyze_unscorable_is_rejected_and_updates_no_mastery(client, monkeypatch):
    monkeypatch.setattr(flask_app, "process_recording", _unscorable_result)
    data = {
        "text": "school",
        "user": "__routes_reject__",
        "audio": (io.BytesIO(b"RIFFfake"), "rec.webm", "audio/webm"),
    }
    resp = client.post("/analyze", data=data, content_type="multipart/form-data").get_json()
    assert resp["scorable"] is False
    assert "record again" in resp["message"].lower()

    # The rejected attempt did not create any mastery-updating evidence.
    import db
    user = db.get_user_by_name("__routes_reject__")
    assert db.get_trusted_recording_count(user["id"]) == 0
    assert db.get_all_phoneme_states(user["id"]) == []
    conn = db.get_connection()
    event_count = conn.execute(
        "SELECT COUNT(*) AS n FROM attempt_phoneme_events e "
        "JOIN attempts a ON a.id=e.attempt_id WHERE a.user_id=?",
        (user["id"],),
    ).fetchone()["n"]
    assert event_count == 0


def test_analyze_response_carries_provenance(client, monkeypatch):
    monkeypatch.setattr(flask_app, "process_recording", _scorable_result)
    data = {
        "text": "school", "user": "__routes_prov__",
        "audio": (io.BytesIO(b"RIFFfake"), "rec.webm", "audio/webm"),
    }
    resp = client.post("/analyze", data=data, content_type="multipart/form-data").get_json()
    assert resp["scoring_engine"] == "panphon"
    import db
    user = db.get_user_by_name("__routes_prov__")
    conn = db.get_connection()
    row = conn.execute("SELECT scoring_engine, scoring_trusted, mastery_updated, "
                       "g2p_mode, reference_g2p_trusted FROM attempts "
                       "WHERE user_id=?", (user["id"],)).fetchone()
    assert row["scoring_engine"] == "panphon"
    assert row["scoring_trusted"] == 1 and row["mastery_updated"] == 1
    assert row["g2p_mode"].startswith("context_aware_")
    assert row["reference_g2p_trusted"] == 1


def test_untrusted_reference_never_updates_mastery(client, monkeypatch):
    monkeypatch.setattr(flask_app, "process_recording", _scorable_result)
    response = client.post(
        "/analyze",
        data={
            "text": "They permit entry",
            "user": "__routes_permit__",
            "audio": (io.BytesIO(b"RIFFfake"), "rec.webm", "audio/webm"),
        },
        content_type="multipart/form-data",
    ).get_json()

    assert response["scorable"] is True
    assert response["scoring_trusted"] is True
    assert response["reference_g2p_trusted"] is False
    assert response["unsupported_heteronyms"] == ["permit"]
    assert response["mastery_updated"] is False
    import db
    user = db.get_user_by_name("__routes_permit__")
    assert db.get_all_phoneme_states(user["id"]) == []
