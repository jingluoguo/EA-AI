from __future__ import annotations

from .structure import Entity


QUESTION_FRAMES = (
    "你知道的话",
    "可以告诉我",
    "你可以告诉我",
    "给我说一下",
    "我想知道",
    "我想问",
    "想知道",
    "想问",
    "想了解",
    "帮我看看",
    "告诉我",
)
QUESTION_FILLERS = ("现在", "请问", "到底", "又", "再", "还", "是", "的", "了", "吗", "嘛", "呢", "下")
CONTAINMENT_VERBS = ("放到", "放入", "放进")
CONTAINER_SUFFIXES = ("里面", "里边", "里头", "内部", "里", "内", "中")


def normalize_question(sentence: str) -> str:
    normalized = sentence.strip().replace("？", "").replace("?", "")
    changed = True
    while changed:
        previous = normalized
        normalized = normalize_containment_expression(strip_question_frames(normalize_slot_value(normalized)))
        changed = normalized != previous
    return normalized


def strip_question_frames(sentence: str) -> str:
    normalized = sentence.strip()
    changed = True
    while changed:
        changed = False
        for frame in sorted(QUESTION_FRAMES, key=len, reverse=True):
            if normalized.startswith(frame) and len(normalized) > len(frame):
                normalized = normalized[len(frame) :]
                changed = True
    return normalized


def normalize_slot_value(value: str) -> str:
    normalized = value.strip()
    changed = True
    while changed:
        changed = False
        for word in sorted(QUESTION_FILLERS, key=len, reverse=True):
            if normalized.startswith(word) and len(normalized) > len(word):
                normalized = normalized[len(word) :]
                changed = True
            if normalized.endswith(word) and len(normalized) > len(word):
                normalized = normalized[: -len(word)]
                changed = True
    return normalized


def normalize_container_slot(value: str) -> str:
    normalized = normalize_slot_value(value)
    changed = True
    while changed:
        changed = False
        for suffix in sorted(CONTAINER_SUFFIXES, key=len, reverse=True):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
    return normalized


def normalize_entity_slot(value: str, entities: tuple[Entity, ...]) -> str:
    normalized = normalize_slot_value(value)
    matches = [entity.name for entity in entities if entity.name in normalized]
    if matches:
        return max(matches, key=len)
    return normalized


def normalize_containment_expression(sentence: str) -> str:
    normalized = sentence
    for verb in CONTAINMENT_VERBS:
        normalized = normalized.replace(verb, "放进")
    return normalized
