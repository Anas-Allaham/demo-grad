"""Audit #2 & #3: engine provenance/trust columns and PanPhon validation."""

import phoneme_vectors_professional as pf


# ---- #3: startup PanPhon validation ----
def test_panphon_validation_consistent_with_trust():
    report = pf.validate_panphon_inventory()
    assert report["checked"] == len(pf.ASSESSABLE_INVENTORY)
    # scoring is trusted iff every assessable phoneme vectorizes.
    assert pf.scoring_trusted() == report["ok"]
    assert pf.scoring_engine() == ("panphon" if report["ok"] else "fallback_features")


def test_incomplete_panphon_is_not_trusted(monkeypatch):
    """If ANY assessable phoneme fails to vectorize, the engine must report
    fallback_features -- never silently 'panphon' + trusted."""
    import phoneme_vectors_professional as p

    real_vector = p.phoneme_vector

    vector_calls = []

    def flaky_vector(ph):
        vector_calls.append(p.canonicalize_phoneme(ph))
        if p.canonicalize_phoneme(ph) == "s":
            raise ValueError("simulated PanPhon gap")
        return real_vector(ph)

    p.validate_panphon_inventory.cache_clear()
    monkeypatch.setattr(p, "phoneme_vector", flaky_vector)
    try:
        report = p.validate_panphon_inventory()
        assert report["ok"] is False
        assert "s" in report["failures"]
        assert p.scoring_engine() == "fallback_features"
        assert p.scoring_trusted() is False
        # Once global validation fails, even pairs PanPhon could vectorize use
        # fallback features. The attempt cannot silently mix engines.
        vector_calls.clear()
        distance = p.articulatory_distance("t", "d")
        assert 0.0 <= distance <= 1.0
        assert vector_calls == []
    finally:
        p.validate_panphon_inventory.cache_clear()  # restore real validation


# ---- #2: trust columns exist and default legacy rows to untrusted ----
def test_trust_columns_present_and_default_untrusted(temp_db):
    db = temp_db
    conn = db.get_connection()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(attempts)").fetchall()}
    for col in (
        "scoring_engine", "scoring_trusted", "mastery_updated", "insertion_count",
        "reference_unit_count", "g2p_mode", "reference_g2p_trusted", "reference_g2p_reason",
    ):
        assert col in cols
    ev_cols = {r["name"] for r in conn.execute("PRAGMA table_info(attempt_phoneme_events)").fetchall()}
    assert "scoring_engine" in ev_cols

    # A legacy-style attempt inserted without trust flags is untrusted (0).
    user = db.get_or_create_user("legacy")
    conn.execute(
        "INSERT INTO attempts (user_id, text, reference_ipa, predicted_ipa, phoneme_error_rate, weighted_error) "
        "VALUES (?, 'x', 'x', 'x', 0, 0)",
        (user["id"],),
    )
    conn.commit()
    row = conn.execute("SELECT scoring_trusted, mastery_updated FROM attempts").fetchone()
    assert row["scoring_trusted"] == 0
    assert row["mastery_updated"] == 0


def test_only_trusted_recordings_count_toward_evidence(temp_db):
    """Assessment coverage / confusion / diagnostic use mastery-updating rows
    only. An untrusted attempt with events must not create phoneme evidence."""
    db = temp_db
    user = db.get_or_create_user("u1")
    uid = user["id"]

    alignment = [
        {"expected": "θ", "spoken": "s", "result": "close_substitution",
         "articulatory_distance": 0.1, "alignment_cost": 0.45},
    ]

    # Untrusted attempt (mastery_updated=0): recorded, but excluded from evidence.
    db.record_attempt(user_id=uid, text="p1", reference_ipa="θ", predicted_ipa="s",
                      phoneme_error_rate=10.0, weighted_error=0.45, scorable=True,
                      scoring_engine="fallback_features", scoring_trusted=False, mastery_updated=False)
    untrusted_id = db.get_user_attempts(uid)[0]["id"]
    db.record_phoneme_events(untrusted_id, alignment)

    assert db.get_confusion_pairs(uid) == []
    assert db.get_phoneme_context_stats(uid) == {}
    assert db.get_trusted_recording_count(uid) == 0

    # A trusted attempt DOES count.
    db.record_recording_atomic(
        user_id=uid,
        attempt_kwargs={"text": "p2", "reference_ipa": "θ", "predicted_ipa": "s",
                        "phoneme_error_rate": 10.0, "weighted_error": 0.45, "scorable": True,
                        "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True},
        alignment=alignment, scoring_engine="panphon",
    )
    assert db.get_trusted_recording_count(uid) == 1
    confusions = db.get_confusion_pairs(uid)
    assert confusions and confusions[0]["expected"] == "θ" and confusions[0]["spoken"] == "s"
    assert "θ" in db.get_phoneme_context_stats(uid)


def test_per_event_engine_provenance_is_stored(temp_db):
    db = temp_db
    user = db.get_or_create_user("prov")
    aid = db.record_attempt(user_id=user["id"], text="t", reference_ipa="s", predicted_ipa="s",
                            phoneme_error_rate=0.0, weighted_error=0.0, scoring_engine="panphon",
                            scoring_trusted=True, mastery_updated=True)
    db.record_phoneme_events(
        aid,
        [{"expected": "s", "spoken": "s", "result": "correct",
          "articulatory_distance": 0.0, "alignment_cost": 0.0}],
    )
    conn = db.get_connection()
    # record_phoneme_events (public) stores NULL engine; the atomic path stores it.
    db.record_recording_atomic(
        user_id=user["id"],
        attempt_kwargs={"text": "t2", "reference_ipa": "s", "predicted_ipa": "s",
                        "phoneme_error_rate": 0.0, "weighted_error": 0.0, "scorable": True,
                        "scoring_engine": "panphon", "scoring_trusted": True, "mastery_updated": True},
        alignment=[{"expected": "s", "spoken": "s", "result": "correct",
                    "articulatory_distance": 0.0, "alignment_cost": 0.0}],
        scoring_engine="panphon",
    )
    engines = {r["scoring_engine"] for r in conn.execute(
        "SELECT scoring_engine FROM attempt_phoneme_events").fetchall()}
    assert "panphon" in engines
