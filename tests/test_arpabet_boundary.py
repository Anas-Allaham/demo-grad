"""IPA stays in the Flask domain; browser/API payloads use ARPAbet."""

from pathlib import Path

import pytest

from arpabet import metrics_to_internal_ipa, to_public_arpabet
from phoneme_alphabet import (
    ARPABET_TO_IPA,
    CANONICAL_ARPABET,
    IPAInputFormat,
    IPA_TO_ARPABET,
    UnsupportedPhonemeError,
    canonicalize_arpabet_phoneme,
    ipa_to_arpabet,
)


def test_boundary_converts_sequences_guides_alignment_and_assessment():
    public = to_public_arpabet({
        "reference_ipa": "s k uː l | ɪ z",
        "predicted_ipa": "s k u l | ɪ s",
        "alignment": [{"expected": "θ", "spoken": "s"}],
        "guide": [{"phonemes": [{"symbol": "u"}]}],
        "assessment": {
            "weak_phonemes": [{"phoneme": "ɹ"}],
            "unknown_phonemes": ["tʃ", "oʊ"],
        },
        "confusion_hint": "θ vs s",
    })

    assert public["reference_arpabet"] == "S K UW L | IH Z"
    assert public["predicted_arpabet"] == "S K UW L | IH S"
    assert public["alignment"] == [{"expected": "TH", "spoken": "S"}]
    assert public["guide"][0]["phonemes"][0]["symbol"] == "UW"
    assert public["assessment"]["weak_phonemes"][0]["phoneme"] == "R"
    assert public["assessment"]["unknown_phonemes"] == ["CH", "OW"]
    assert public["confusion_hint"] == "TH vs S"
    assert "reference_ipa" not in public and "predicted_ipa" not in public


def test_public_metrics_are_converted_to_internal_ipa():
    assert metrics_to_internal_ipa({"TH1": 0.2, "S": 0.8}) == {
        "θ": 0.2,
        "s": 0.8,
    }


def test_public_metrics_reject_unknown_arpabet():
    with pytest.raises(UnsupportedPhonemeError):
        metrics_to_internal_ipa({"BAD": 0.4})


def test_public_inventory_is_complete_and_consistent():
    assert set(ARPABET_TO_IPA) == CANONICAL_ARPABET
    assert set(IPA_TO_ARPABET.values()) <= CANONICAL_ARPABET


def test_residual_length_mark_is_canonicalized_before_public_output():
    assert ipa_to_arpabet(
        "b ɝː d z | .",
        input_format=IPAInputFormat.FORMATTED_REFERENCE,
    ) == "B ER D Z"


@pytest.mark.parametrize("value", ["?", "XYZ", "tʰ", "n̩", "æ̃"])
def test_unknown_or_unsupported_phonemes_are_rejected(value):
    with pytest.raises(UnsupportedPhonemeError):
        canonicalize_arpabet_phoneme(value)


def test_reference_and_ctc_word_boundaries_are_explicit():
    value = "aɪ oʊ"
    assert ipa_to_arpabet(
        value,
        input_format=IPAInputFormat.FORMATTED_REFERENCE,
    ) == "AY OW"
    assert ipa_to_arpabet(
        value,
        input_format=IPAInputFormat.RAW_CTC,
    ) == "AY | OW"


def test_non_strict_ctc_recovery_drops_and_logs_unknown_symbols(caplog):
    assert ipa_to_arpabet(
        "a e o ? s[UNK]kuːl",
        input_format=IPAInputFormat.RAW_CTC,
        strict=False,
    ) == "AA | EH | OW | S K UW L"
    assert "Dropping unsupported phoneme" in caplog.text
    assert "Dropping tokenizer control token" in caplog.text


@pytest.mark.parametrize("arpabet", sorted(CANONICAL_ARPABET))
def test_canonical_arpabet_round_trip(arpabet):
    assert canonicalize_arpabet_phoneme(ARPABET_TO_IPA[arpabet]) == arpabet


def test_database_schema_remains_ipa(temp_db):
    columns = {
        row["name"] for row in temp_db.get_connection().execute("PRAGMA table_info(attempts)")
    }
    assert {"reference_ipa", "predicted_ipa"} <= columns
    assert "reference_arpabet" not in columns


def test_browser_uses_arpabet_response_fields():
    root = Path(__file__).resolve().parent.parent
    script = (root / "static" / "script.js").read_text(encoding="utf-8")
    template = (root / "templates" / "index.html").read_text(encoding="utf-8")
    assert "data.reference_arpabet" in script
    assert "data.predicted_arpabet" in script
    assert "data.reference_ipa" not in script
    assert "data.predicted_ipa" not in script
    assert 'id="referenceArpabet"' in template
    assert 'id="predictedArpabet"' in template
    assert "ARPAbet symbol" in template
