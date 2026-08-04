from __future__ import annotations

import re


def split_sentences(text: str) -> tuple[tuple[str, bool], ...]:
    stripped = text.strip()
    parts: list[tuple[str, bool]] = []
    last_end = 0
    for match in re.finditer(r"([^。？！?]+)([。？！?])", stripped):
        sentence = match.group(1).strip(" ，,")
        if sentence:
            parts.append((sentence, match.group(2) in "？?"))
        last_end = match.end()

    tail = stripped[last_end:].strip(" ，,")
    if tail:
        parts.append((tail, False))

    return tuple(parts)
