from __future__ import annotations

from ..structure import Entity


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
QUESTION_FILLERS = (
    "现在",
    "目前",
    "实际",
    "事实上",
    "后来",
    "之后",
    "随后",
    "请问",
    "到底",
    "其实",
    "又",
    "再",
    "还",
    "是",
    "的",
    "了",
    "吗",
    "嘛",
    "呢",
    "下",
)
QUESTION_INNER_FILLERS = (
    "现在",
    "目前",
    "实际",
    "事实上",
    "请问",
    "到底",
    "其实",
)
QUESTION_SURFACE_SYNONYMS = (
    ("物品", "东西"),
)
CONTAINMENT_VERBS = ("放到", "放入", "放进")
TAKE_OUT_VERBS = ("取出来", "拿出来", "取出", "拿出", "取走", "拿走")
CONTAINER_SUFFIXES = ("里面", "里边", "里头", "内部", "里", "内", "中")
CONTAINER_SURFACE_SUFFIXES = ("里面", "里边", "里头", "内部")
DEMONSTRATIVE_PREFIXES = ("这个", "这件", "那个", "那件")


def normalize_question(sentence: str) -> str:
    normalized = sentence.strip().replace("？", "").replace("?", "")
    normalized = normalized.replace("啥", "什么")
    normalized = normalize_question_surface_words(normalized)
    changed = True
    while changed:
        previous = normalized
        normalized = normalize_take_out_expression(
            normalize_containment_expression(strip_question_frames(normalize_slot_value(normalized)))
        )
        changed = normalized != previous
    return normalized


def normalize_question_surface_words(sentence: str) -> str:
    normalized = sentence
    for source, target in QUESTION_SURFACE_SYNONYMS:
        normalized = normalized.replace(source, target)
    for word in QUESTION_INNER_FILLERS:
        normalized = normalized.replace(word, "")
    return normalized


def strip_question_frames(sentence: str) -> str:
    normalized = sentence.strip()
    changed = True
    while changed:
        changed = False
        for frame in sorted(QUESTION_FRAMES, key=len, reverse=True):
            if normalized == frame:
                return ""
            if normalized.startswith(frame) and len(normalized) > len(frame):
                normalized = normalized[len(frame) :]
                changed = True
    return normalized


def normalize_slot_value(value: str) -> str:
    normalized = value.strip()
    changed = True
    while changed:
        changed = False
        for prefix in sorted(DEMONSTRATIVE_PREFIXES, key=len, reverse=True):
            if normalized.startswith(prefix) and len(normalized) > len(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
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


def normalize_container_surface(sentence: str) -> str:
    normalized = sentence
    for suffix in CONTAINER_SURFACE_SUFFIXES:
        normalized = normalized.replace(suffix, "里")
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


def normalize_take_out_expression(sentence: str) -> str:
    normalized = sentence
    for verb in TAKE_OUT_VERBS:
        normalized = normalized.replace(verb, "取出")
    return normalized

