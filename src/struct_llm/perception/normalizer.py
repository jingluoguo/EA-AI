from __future__ import annotations

import unicodedata

from ..comprehension.surface_lexicon import surface_forms
from ..structure import Entity


def normalize_question(sentence: str) -> str:
    from ..neural.perception_classifier import default_perception_model

    return default_perception_model().normalize(sentence, "question")


def normalize_statement(sentence: str) -> str:
    from ..neural.perception_classifier import default_perception_model

    return default_perception_model().normalize(sentence, "statement")


def normalize_slot_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = strip_edge_noise(normalized)
    changed = True
    while changed:
        changed = False
        for prefix in sorted(surface_forms("demonstrative_prefix"), key=len, reverse=True):
            if normalized.startswith(prefix) and len(normalized) > len(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
        for word in sorted(surface_forms("question_filler"), key=len, reverse=True):
            if normalized.startswith(word) and len(normalized) > len(word):
                normalized = normalized[len(word) :]
                changed = True
            if normalized.endswith(word) and len(normalized) > len(word):
                normalized = normalized[: -len(word)]
                changed = True
    return strip_edge_noise(normalized)


def strip_edge_noise(text: str) -> str:
    stripped = text.strip()
    while stripped and is_edge_noise(stripped[0]):
        stripped = stripped[1:].lstrip()
    while stripped and is_edge_noise(stripped[-1]):
        stripped = stripped[:-1].rstrip()
    return stripped


def is_edge_noise(char: str) -> bool:
    category = unicodedata.category(char)
    return char.isspace() or category in {"Cc", "Cf"} or category.startswith(("P", "S"))


def normalize_container_slot(value: str) -> str:
    normalized = normalize_slot_value(value)
    changed = True
    while changed:
        changed = False
        for suffix in sorted(surface_forms("container_suffix"), key=len, reverse=True):
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
