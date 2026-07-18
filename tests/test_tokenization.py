"""Tests 1-4: tokenization + canonical mapping."""

from tokenization import (
    ipa_to_tokens,
    tokenize_ctc_prediction,
    tokenize_reference_ipa,
)
from phoneme_vectors_professional import canonicalize_phoneme


# 1. Multiword CTC tokenization: whole words must split into phonemes.
def test_multiword_ctc_tokenization():
    assert tokenize_ctc_prediction("skuːl ɪz oʊpən") == [
        "s", "k", "u", "l", "ɪ", "z", "oʊ", "p", "ə", "n"
    ]


# 2. Formatted reference tokenization: already phoneme-spaced with | words.
def test_formatted_reference_tokenization():
    assert tokenize_reference_ipa("s k uː l | ɪ z | oʊ p ə n") == [
        "s", "k", "u", "l", "ɪ", "z", "oʊ", "p", "ə", "n"
    ]


def test_both_formats_agree():
    ctc = tokenize_ctc_prediction("skuːl ɪz oʊpən")
    ref = tokenize_reference_ipa("s k uː l | ɪ z | oʊ p ə n")
    assert ctc == ref
    assert ipa_to_tokens("skuːl ɪz oʊpən") == ref


# 3. Canonical long-vowel mapping: model has no /ː/, so long vowels collapse.
def test_canonical_long_vowel_mapping():
    assert canonicalize_phoneme("iː") == "i"
    assert canonicalize_phoneme("uː") == "u"
    assert canonicalize_phoneme("ɑː") == "ɑ"
    assert canonicalize_phoneme("ɔː") == "ɔ"
    # Tokenization applies the same mapping.
    assert tokenize_reference_ipa("f iː l") == ["f", "i", "l"]
    assert tokenize_ctc_prediction("fuːd") == ["f", "u", "d"]


# 4. Composite affricate/diphthong parsing.
def test_composite_affricate_and_diphthong_parsing():
    assert tokenize_ctc_prediction("tʃɛr") == ["tʃ", "ɛ", "ɹ"]
    assert tokenize_ctc_prediction("dʒʌmp") == ["dʒ", "ʌ", "m", "p"]
    assert tokenize_ctc_prediction("maɪ") == ["m", "aɪ"]
    assert tokenize_ctc_prediction("naʊ") == ["n", "aʊ"]
    assert tokenize_ctc_prediction("ɡoʊ") == ["ɡ", "oʊ"]
    assert tokenize_ctc_prediction("bɔɪ") == ["b", "ɔɪ"]
    assert tokenize_ctc_prediction("seɪ") == ["s", "eɪ"]
