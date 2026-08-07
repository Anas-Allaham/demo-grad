"""Predictive ByT5 OOV fallback.

A word with no reviewed lexicon entry that the predictor CAN pronounce becomes
an untrusted ``predicted`` word: it carries a pronunciation (so it is not left
as a raw-grapheme ``oov`` fallback) but keeps the reference untrusted, so it
never feeds mastery or the trusted exercise bank. A word the predictor cannot
handle (or when the predictor is absent) stays a hard ``oov``.

These tests inject a fake predictor and never load transformer weights. The
real model is exercised only by the opt-in integration test at the bottom.
"""

from __future__ import annotations

import os

import pytest

from g2p_service import DictionaryIpaG2p, ReferenceG2PResult
from oov_g2p import ByT5OOVPredictor

# Letters only (so the tokenizer extracts it as one word), absent from the
# bundled CMU-derived dictionary and not a heteronym.
OOV_WORD = "zzxqwlpr"


class FakePredictor:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def predict(self, word):
        self.calls.append(word)
        return self.mapping.get(word.lower(), "")


def test_predicted_oov_is_untrusted_but_not_raw_grapheme():
    predictor = FakePredictor({OOV_WORD: "z æ p"})
    engine = DictionaryIpaG2p(oov_predictor=predictor)

    result = engine.resolve(OOV_WORD)

    assert OOV_WORD in result.predicted_words
    assert result.oov_words == ()
    assert result.reference_g2p_trusted is False
    assert result.ipa  # a real pronunciation was produced
    assert predictor.calls == [OOV_WORD]


def test_oov_without_predictor_stays_hard_oov():
    engine = DictionaryIpaG2p(oov_predictor=None)

    result = engine.resolve(OOV_WORD)

    assert OOV_WORD in result.oov_words
    assert result.predicted_words == ()
    assert result.reference_g2p_trusted is False


def test_empty_prediction_falls_back_to_oov():
    predictor = FakePredictor({})  # predict() -> "" for everything
    engine = DictionaryIpaG2p(oov_predictor=predictor)

    result = engine.resolve(OOV_WORD)

    assert OOV_WORD in result.oov_words
    assert result.predicted_words == ()


def test_known_word_never_reaches_predictor():
    predictor = FakePredictor({})
    engine = DictionaryIpaG2p(oov_predictor=predictor)

    result = engine.resolve("hello")

    assert result.reference_g2p_trusted is True
    assert result.oov_words == ()
    assert result.predicted_words == ()
    assert predictor.calls == []  # dictionary hit short-circuits the predictor


def test_predicted_word_present_in_to_dict():
    predictor = FakePredictor({OOV_WORD: "z æ p"})
    engine = DictionaryIpaG2p(oov_predictor=predictor)

    payload = engine.resolve(OOV_WORD).to_dict()

    assert payload["predicted_words"] == [OOV_WORD]
    assert payload["reference_g2p_trusted"] is False


def test_to_inventory_ipa_keeps_english_drops_foreign():
    # No model load: _to_inventory_ipa is a pure canonicalize+filter step.
    predictor = ByT5OOVPredictor()
    cleaned = predictor._to_inventory_ipa("t͡ʃ æ t ǃ")  # tie-bar affricate + a click
    tokens = cleaned.split()

    assert "tʃ" in tokens  # tie-bar affricate canonicalized into the inventory
    assert "æ" in tokens and "t" in tokens
    assert "ǃ" not in cleaned  # out-of-inventory symbol dropped


@pytest.mark.skipif(
    not os.environ.get("OOV_G2P_MODEL_TEST"),
    reason="opt-in: set OOV_G2P_MODEL_TEST=1 to load the real ByT5 weights",
)
def test_real_byt5_predicts_chatgpt():
    from phoneme_vectors_professional import in_inventory

    predictor = ByT5OOVPredictor()
    ipa = predictor.predict("chatgpt")

    assert ipa  # a non-empty, in-inventory pronunciation
    assert all(in_inventory(tok) for tok in ipa.split())
