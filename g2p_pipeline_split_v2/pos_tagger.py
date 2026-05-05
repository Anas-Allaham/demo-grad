from typing import List, Tuple


class PosTagger:
    def tag(self, text: str) -> List[Tuple[str, str]]:
        raise NotImplementedError


class SpacyPosTagger(PosTagger):
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        import spacy
        self.spacy = spacy
        self.model_name = model_name
        self.nlp = self._load_model()

    def _load_model(self):
        try:
            return self.spacy.load(self.model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{self.model_name}' is not installed.\n"
                f"Install it with:\n"
                f"python -m spacy download {self.model_name}"
            )

    def tag(self, text: str) -> List[Tuple[str, str]]:
        doc = self.nlp(text)
        return [(token.text, token.tag_) for token in doc if not token.is_space]
