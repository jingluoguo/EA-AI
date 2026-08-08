from __future__ import annotations

def split_sentences(text: str) -> tuple[tuple[str, bool], ...]:
    stripped = text.strip()
    parts: list[tuple[str, bool]] = []
    start = 0
    for index, character in enumerate(stripped):
        if character not in "。？！?":
            continue
        sentence = stripped[start:index].strip(" ，,")
        if sentence:
            parts.append((sentence, character in "？?"))
        start = index + 1

    tail = stripped[start:].strip(" ，,")
    if tail:
        parts.append((tail, False))

    return tuple(parts)


def split_query_candidate(candidate: str) -> tuple[str, ...]:
    normalized = candidate.strip().rstrip("。！？!?")
    parts: list[str] = []
    start = 0
    for index, character in enumerate(normalized):
        if character not in "，,；;":
            continue
        fragment = normalized[start:index].strip()
        if not fragment:
            start = index + 1
            continue
        fragment = fragment.strip("，,；;")
        if fragment:
            parts.append(fragment)
        start = index + 1
    tail = normalized[start:].strip("，,；;")
    if tail:
        parts.append(tail)
    return tuple(parts) if parts else (normalized,)
