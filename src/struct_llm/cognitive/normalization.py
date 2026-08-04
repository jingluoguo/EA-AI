from __future__ import annotations

import re

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
CONTAINMENT_VERBS = ("放到", "放入", "放进")
TAKE_OUT_VERBS = ("取出来", "拿出来", "取出", "拿出", "取走", "拿走")
CONTAINER_SUFFIXES = ("里面", "里边", "里头", "内部", "里", "内", "中")
DEMONSTRATIVE_PREFIXES = ("这个", "这件", "那个", "那件")
QUESTION_NOISE_RE = re.compile(
    r"^(?:你|您)?(?:知道|了解|明白)(?:吗|嘛)?(?:的话)?$|^(?:请问|告诉我|帮我看看|给我说一下|想知道|想了解|想问)$"
)
CHAT_EXPRESSION_HINTS = (
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "谢谢",
    "感谢",
    "再见",
    "拜拜",
    "你是谁",
    "你叫什么",
    "你能做什么",
    "你会什么",
    "你可以做什么",
    "总结",
    "概括",
    "回顾",
    "刚才说了什么",
    "我叫什么",
    "我是谁",
    "我喜欢什么",
    "我讨厌什么",
)


def normalize_question(sentence: str) -> str:
    normalized = sentence.strip().replace("？", "").replace("?", "")
    changed = True
    while changed:
        previous = normalized
        normalized = normalize_take_out_expression(
            normalize_containment_expression(strip_question_frames(normalize_slot_value(normalized)))
        )
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


def is_question_noise(sentence: str) -> bool:
    normalized = normalize_question(sentence).strip()
    if not normalized:
        return True
    if any(hint in normalized for hint in CHAT_EXPRESSION_HINTS):
        return False
    if QUESTION_NOISE_RE.match(normalized):
        return True
    if len(normalized) <= 3 and not any(
        word in normalized for word in ("谁", "什么", "哪里", "哪儿", "哪", "有", "在", "是", "几", "多", "最", "前", "后")
    ):
        return True
    return False
