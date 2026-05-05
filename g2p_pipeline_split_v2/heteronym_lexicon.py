import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import normalize_candidate


class ExternalHeteronymLexicon:
    def __init__(self, json_path: str) -> None:
        self.json_path = Path(json_path)
        self.entries = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.lower(): v for k, v in data.items()}

    def contains(self, token: str) -> bool:
        return token.lower() in self.entries

    def resolve(self, token: str, pos_tag: str) -> Optional[List[str]]:
        entry = self.entries.get(token.lower())
        if not entry:
            return None

        pos_map = entry.get("pos_map", {})
        if pos_tag in pos_map:
            return normalize_candidate(pos_map[pos_tag])

        default = entry.get("default")
        if default is not None:
            return normalize_candidate(default)

        return None
