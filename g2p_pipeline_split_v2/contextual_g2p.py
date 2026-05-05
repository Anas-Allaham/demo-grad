from typing import List, Optional

from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p

from heteronym_lexicon import ExternalHeteronymLexicon
from pos_tagger import PosTagger, SpacyPosTagger


class ContextAwareIpaG2p:
    """
    Pipeline:
    raw text -> spaCy POS tagging -> external heteronym lexicon (IPA) -> NeMo IpaG2p fallback
    """

    def __init__(
        self,
        heteronyms_json_path: str,
        ipa_dict_path: str,
        g2p: Optional[IpaG2p] = None,
        tagger: Optional[PosTagger] = None,
    ) -> None:
        self.ipa_dict_path = str(ipa_dict_path)

        self.g2p = g2p or IpaG2p(
            phoneme_dict=self.ipa_dict_path,
            locale="en-US",
            ignore_ambiguous_words=False,
            use_chars=False,
            use_stresses=True,
        )

        self.tagger = tagger or SpacyPosTagger("en_core_web_sm")
        self.lexicon = ExternalHeteronymLexicon(heteronyms_json_path)

    def _normalize_ipa_output(self, ipa_out) -> str:
        """
        Keeps only the first pronunciation from NeMo output
        and returns it as one IPA string like: 'ɹɛd'
        """
        if isinstance(ipa_out, str):
            ipa_out = [ipa_out]
        elif not isinstance(ipa_out, list):
            ipa_out = [str(ipa_out)]

        cleaned = []
        for item in ipa_out:
            if item == ",":
                break
            cleaned.append(item)

        return "".join(cleaned)

    def __call__(self, text: str) -> List[str]:
        tagged = self.tagger.tag(text)
        output: List[str] = []

        for token, pos_tag in tagged:
            resolved = self.lexicon.resolve(token, pos_tag)

            if resolved is not None:
                output.append("".join(resolved))
            else:
                ipa_out = self.g2p(token)
                output.append(self._normalize_ipa_output(ipa_out))

        return output