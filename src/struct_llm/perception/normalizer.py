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
QUESTION_NOISE_PHRASES = (
    "请问",
    "告诉我",
    "帮我看看",
    "给我说一下",
    "想知道",
    "想了解",
    "想问",
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


def is_question_noise(sentence: str) -> bool:
    normalized = normalize_question(sentence).strip()
    if not normalized:
        return True
    if any(hint in normalized for hint in CHAT_EXPRESSION_HINTS):
        return False
    if normalized in QUESTION_NOISE_PHRASES or normalized in {
        "知道",
        "知道吗",
        "知道嘛",
        "知道的话",
        "了解",
        "了解吗",
        "了解嘛",
        "了解的话",
        "明白",
        "明白吗",
        "明白嘛",
        "明白的话",
        "你知道",
        "你知道吗",
        "你知道嘛",
        "你知道的话",
        "你了解",
        "你了解吗",
        "你了解嘛",
        "你了解的话",
        "你明白",
        "你明白吗",
        "你明白嘛",
        "你明白的话",
        "您知道",
        "您知道吗",
        "您知道嘛",
        "您知道的话",
        "您了解",
        "您了解吗",
        "您了解嘛",
        "您了解的话",
        "您明白",
        "您明白吗",
        "您明白嘛",
        "您明白的话",
    }:
        return True
    if len(normalized) <= 3 and not any(
        word in normalized for word in ("谁", "什么", "哪里", "哪儿", "哪", "有", "在", "是", "几", "多", "最", "前", "后")
    ):
        return True
    return False
