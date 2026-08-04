from __future__ import annotations

import re


QUERY_HINTS = (
    "谁",
    "什么",
    "哪里",
    "哪儿",
    "什么地方",
    "为什么",
    "为啥",
    "为何",
    "怎么",
    "几个",
    "多少",
    "数量",
    "是否",
    "是不是",
    "有没有",
    "会不会",
    "能不能",
    "可不可以",
    "还是",
    "吗",
    "嘛",
    "呢",
)


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


def split_query_candidate(candidate: str) -> tuple[str, ...]:
    normalized = candidate.strip().rstrip("。！？!?")
    parts = []
    for raw in re.split(r"[，,；;]", normalized):
        fragment = raw.strip()
        if not fragment:
            continue
        fragment = fragment.strip("，,；;")
        if fragment:
            parts.append(fragment)
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


def is_query_like_fragment(fragment: str) -> bool:
    normalized = fragment.strip().rstrip("。！？!?")
    return any(hint in normalized for hint in QUERY_HINTS)


def should_merge_with_next_fragment(fragment: str, next_fragment: str) -> bool:
    normalized = fragment.strip()
    next_normalized = next_fragment.strip()
    if normalized.startswith("如果") and "没有" in normalized:
        return True
    if "没有" in normalized and any(word in next_normalized for word in ("会在哪里", "会在哪儿", "会在什么地方")):
        return True
    if "之前" in normalized and any(word in normalized for word in ("哪里", "哪儿", "什么地方")):
        return True
    if "之后" in normalized and "发生" in normalized and "什么" in normalized:
        return True
    return False
