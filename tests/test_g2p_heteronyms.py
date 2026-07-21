"""Context-aware heteronyms must work with the dictionary-only backend."""

import pytest

from g2p_service import DictionaryIpaG2p, validate_heteronym_lexicon
from tokenization import tokenize_reference_ipa


def _word_ipa(result, index):
    return result.ipa.split("|")[index].strip().replace(" ", "")


@pytest.mark.parametrize(
    "word,text_a,index_a,ipa_a,text_b,index_b,ipa_b",
    [
        ("read", "I read every day", 1, "ɹid", "I read it yesterday", 1, "ɹɛd"),
        ("record", "They record music", 1, "ɹɪkɔɹd", "The record was broken", 1, "ɹɛkɝd"),
        ("present", "Please present the report", 1, "pɹɪzɛnt", "The present is here", 1, "pɹɛzənt"),
        ("use", "I use this tool", 1, "juz", "The use is clear", 1, "jus"),
        ("refuse", "They refuse the offer", 1, "ɹɪfjuz", "The refuse was collected", 1, "ɹɛfjus"),
        ("close", "Please close the door", 1, "kloʊz", "Stay close to me", 1, "kloʊs"),
    ],
)
def test_dictionary_fallback_resolves_context(word, text_a, index_a, ipa_a, text_b, index_b, ipa_b):
    engine = DictionaryIpaG2p()
    result_a = engine.resolve(text_a)
    result_b = engine.resolve(text_b)

    assert _word_ipa(result_a, index_a) == ipa_a
    assert _word_ipa(result_b, index_b) == ipa_b
    assert result_a.reference_g2p_trusted is True
    assert result_b.reference_g2p_trusted is True
    assert result_a.g2p_mode == "context_aware_dictionary_fallback"
    assert result_a.heteronym_resolution_active is True


def test_permit_contrast_is_explicitly_unsupported_for_mastery():
    engine = DictionaryIpaG2p()
    verb = engine.resolve("They permit entry")
    noun = engine.resolve("The permit expired")

    assert _word_ipa(verb, 1) == "pɚmɪt"
    assert _word_ipa(noun, 1) == "pɝmɪt"
    assert tokenize_reference_ipa(_word_ipa(verb, 1)) == tokenize_reference_ipa(_word_ipa(noun, 1))
    assert verb.reference_g2p_trusted is False
    assert noun.reference_g2p_trusted is False
    assert verb.unsupported_heteronyms == ("permit",)


def test_all_72_heteronym_entries_validate_and_permit_is_reported():
    report = validate_heteronym_lexicon()
    assert report["checked"] == 72
    assert report["valid_entries"] == 72
    assert report["schema_and_inventory_ok"] is True
    assert report["invalid_entries"] == []
    assert report["unsupported_phonemes"] == {}
    assert report["unsupported_contrasts"] == ["permit"]
    assert report["fully_supported"] is False
