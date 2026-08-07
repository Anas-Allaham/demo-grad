"""
Predictive out-of-vocabulary (OOV) G2P using a pretrained ByT5 model.

The trusted G2P path (the bundled CMU-derived dictionary, or NeMo ``IpaG2p``
when it is installed) can only return a pronunciation for a word that has a
reviewed dictionary entry. For a genuinely out-of-vocabulary spelling (coined
product names, blends, brand initialisms, ...) the dictionary misses and NeMo
just echoes the graphemes back -- neither is a real pronunciation.

This module adds a *predictive* fallback: a pretrained ByT5 grapheme->IPA model
(the multilingual CharsiuG2P checkpoints) that generates a plausible
pronunciation for an arbitrary spelling.

TRUST BOUNDARY
--------------
A predicted pronunciation is a MODEL GUESS, not a reviewed lexicon entry. It is
therefore always UNTRUSTED: usable for display and practice, but it must never
become mastery evidence nor seed the trusted exercise bank. The G2P service
marks any word served from this predictor as ``predicted`` (not ``oov``), which
keeps the reference ``reference_g2p_trusted == False``.

AVAILABILITY
------------
The predictor is loaded lazily and only when ``OOV_G2P_ENABLED`` is truthy. If
``transformers``/``torch`` or the weights are unavailable, ``load_oov_predictor``
returns ``None`` and the service behaves exactly as before (OOV words keep their
raw-grapheme fallback and stay untrusted). Nothing here reaches the network at
request time when the weights are already cached in the image/local HF cache.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Dict, Optional

from phoneme_vectors_professional import in_inventory
from tokenization import tokenize_reference_ipa

# Small, CPU-friendly multilingual ByT5 G2P checkpoint. The deploy image runs on
# CPU with a 4 GB memory ceiling, so the "tiny" checkpoint is the default;
# override with OOV_G2P_MODEL for the larger "small" checkpoint on a roomier
# host.
DEFAULT_MODEL = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
# CharsiuG2P language identifier for US English; the model was trained on inputs
# of the shape "<eng-us>: word".
DEFAULT_LANG_TAG = "<eng-us>"


def oov_g2p_enabled() -> bool:
    """Whether the predictive OOV fallback should be constructed at all."""
    value = os.environ.get("OOV_G2P_ENABLED", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


class ByT5OOVPredictor:
    """Pretrained ByT5 grapheme->IPA model, filtered to the scoring inventory.

    ``predict`` returns a *canonical, in-inventory* spaced-IPA string so the
    result is guaranteed to tokenize and to survive the strict IPA->ARPAbet
    conversion at the public API boundary. Phonemes the model emits that fall
    outside the app's English scoring inventory are dropped; if nothing scorable
    remains, ``predict`` returns ``""`` and the caller keeps the word as OOV.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        lang_tag: Optional[str] = None,
        max_length: int = 64,
    ) -> None:
        self.model_name = model_name or os.environ.get("OOV_G2P_MODEL", DEFAULT_MODEL)
        self.lang_tag = lang_tag or os.environ.get("OOV_G2P_LANG_TAG", DEFAULT_LANG_TAG)
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._load_lock = Lock()
        self._cache: Dict[str, str] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            model.eval()
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                # CharsiuG2P checkpoints are ByT5; the byte tokenizer is
                # algorithmic and needs no files or network. Fall back to it
                # directly rather than fetching another repo.
                from transformers import ByT5Tokenizer

                tokenizer = ByT5Tokenizer()
            self._tokenizer = tokenizer
            self._model = model

    def _generate_raw_ipa(self, word: str) -> str:
        import torch

        self._ensure_loaded()
        prompt = f"{self.lang_tag}: {word}"
        encoding = self._tokenizer(
            [prompt], padding=True, add_special_tokens=False, return_tensors="pt"
        )
        with torch.no_grad():
            preds = self._model.generate(
                **encoding, num_beams=1, max_length=self.max_length
            )
        decoded = self._tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def _to_inventory_ipa(self, raw_ipa: str) -> str:
        """Canonicalize model IPA and keep only phonemes in the scoring
        inventory, returned as a phoneme-spaced word."""
        # Tie bars (U+0361/U+035C) sit between the two letters of an affricate;
        # strip them so "t͡ʃ"/"d͡ʒ" are recognized as single inventory phonemes
        # (tʃ/dʒ) instead of splitting into their component segments.
        raw_ipa = raw_ipa.replace("͡", "").replace("͜", "")
        tokens = [tok for tok in tokenize_reference_ipa(raw_ipa) if in_inventory(tok)]
        return " ".join(tokens)

    def predict(self, word: str) -> str:
        """Return canonical in-inventory spaced IPA for ``word``, or ``""``.

        Deterministic (greedy decoding) and memoized per word.
        """
        key = (word or "").strip().lower()
        if not key:
            return ""
        if key in self._cache:
            return self._cache[key]
        try:
            ipa = self._to_inventory_ipa(self._generate_raw_ipa(key))
        except Exception:
            ipa = ""
        self._cache[key] = ipa
        return ipa


_predictor_singleton: Optional[ByT5OOVPredictor] = None
_predictor_resolved = False
_resolve_lock = Lock()


def load_oov_predictor(warm: bool = True) -> Optional[ByT5OOVPredictor]:
    """Return the process-wide OOV predictor, or ``None`` when unavailable.

    Resolved once per process. When ``warm`` is true the model weights are
    loaded eagerly so startup fails fast (or degrades to ``None``) rather than
    paying the load cost -- and risking a surprise failure -- on the first
    request. Any failure to construct/load the predictor is logged and treated
    as "unavailable", leaving the previous OOV behaviour intact.
    """
    global _predictor_singleton, _predictor_resolved
    if _predictor_resolved:
        return _predictor_singleton
    with _resolve_lock:
        if _predictor_resolved:
            return _predictor_singleton
        _predictor_resolved = True
        if not oov_g2p_enabled():
            _predictor_singleton = None
            return None
        try:
            predictor = ByT5OOVPredictor()
            if warm:
                predictor._ensure_loaded()
            _predictor_singleton = predictor
        except Exception as exc:  # pragma: no cover - env dependent
            print(
                "Predictive OOV G2P unavailable; out-of-vocabulary words keep "
                "their previous untrusted fallback. Reason:",
                repr(exc),
            )
            _predictor_singleton = None
    return _predictor_singleton
