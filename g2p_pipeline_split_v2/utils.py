from typing import Any, List


def normalize_candidate(candidate: Any) -> List[str]:
    if isinstance(candidate, str):
        return candidate.split()

    if isinstance(candidate, (list, tuple)):
        out: List[str] = []
        for item in candidate:
            if isinstance(item, str):
                if " " in item:
                    out.extend(item.split())
                else:
                    out.append(item)
            else:
                out.append(str(item))
        return out

    return str(candidate).split()
