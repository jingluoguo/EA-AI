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
    if not parts:
        return (normalized,)

    merged: list[str] = []
    index = 0
    while index < len(parts):
        fragment = parts[index]
        next_fragment = parts[index + 1] if index + 1 < len(parts) else ""
        if index + 1 < len(parts) and should_merge_with_next_fragment(fragment, next_fragment):
            merged.append(f"{fragment}，{parts[index + 1]}")
            index += 2
            continue
        merged.append(fragment)
        index += 1

    return tuple(merged)


def should_merge_with_next_fragment(fragment: str, next_fragment: str) -> bool:
    normalized = fragment.strip()
    next_normalized = next_fragment.strip()
    if normalized.startswith("如果") and "就" in next_normalized:
        return True
    if normalized.startswith("因为") and next_normalized.startswith("所以"):
        return True
    if normalized.startswith("如果") and "没有" in normalized:
        return True
    if "没有" in normalized and any(word in next_normalized for word in ("会在哪里", "会在哪儿", "会在什么地方")):
        return True
    if "不是在" in normalized and next_normalized.startswith("是"):
        return True
    if "之前" in normalized and any(word in normalized for word in ("哪里", "哪儿", "什么地方")):
        return True
    if "之后" in normalized and "发生" in normalized and "什么" in normalized:
        return True
    return False
