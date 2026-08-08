from __future__ import annotations

from ..comprehension.surface_lexicon import surface_forms, surface_replacements
from ..structure import Entity


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


def bare_topic_followup(sentence: str) -> str | None:
    raw = sentence.strip()
    has_question_mark = raw.endswith(("？", "?"))
    stripped = raw.rstrip("。！？!?，,；;")
    particles = tuple(sorted(surface_forms("bare_topic_particle"), key=len, reverse=True))
    has_topic_particle = bool(particles) and any(stripped.endswith(particle) for particle in particles)
    if not has_topic_particle and not has_question_mark:
        return None
    if has_question_mark and not has_topic_particle:
        terminal_particles = tuple(surface_forms("terminal_discourse_particle"))
        if any(stripped.endswith(particle) for particle in terminal_particles):
            return None
        core = raw[:-1].strip()
        if any(separator in core for separator in ("。", "！", "!", "？", "?", "；", ";", "，", ",")):
            return None
    topic = stripped
    if has_topic_particle:
        particle = next(particle for particle in particles if stripped.endswith(particle))
        topic = stripped[: -len(particle)].strip()
    if not topic or any(separator in topic for separator in ("，", ",", "；", ";")):
        return None
    normalized = normalize_slot_value(topic)
    if not normalized or normalized in surface_forms("bare_reference_word"):
        return None
    if any(marker in normalized for marker in surface_forms("query_marker")):
        return None
    return normalized


def normalize_question_surface_words(sentence: str) -> str:
    normalized = sentence
    for source, target in surface_replacements("question_surface_synonym"):
        normalized = normalized.replace(source, target)
    for word in surface_forms("question_inner_filler"):
        normalized = normalized.replace(word, "")
    return normalized


def strip_question_frames(sentence: str) -> str:
    normalized = sentence.strip()
    changed = True
    while changed:
        changed = False
        for frame in sorted(surface_forms("question_frame"), key=len, reverse=True):
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
    return normalized


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


def normalize_container_surface(sentence: str) -> str:
    normalized = sentence
    for source, target in surface_replacements("container_surface_suffix"):
        normalized = normalized.replace(source, target)
    return normalized


def normalize_entity_slot(value: str, entities: tuple[Entity, ...]) -> str:
    normalized = normalize_slot_value(value)
    matches = [entity.name for entity in entities if entity.name in normalized]
    if matches:
        return max(matches, key=len)
    return normalized


def normalize_containment_expression(sentence: str) -> str:
    normalized = sentence
    for source, target in surface_replacements("containment_verb"):
        normalized = normalized.replace(source, target)
    return normalized


def normalize_take_out_expression(sentence: str) -> str:
    normalized = sentence
    for source, target in surface_replacements("take_out_verb"):
        normalized = normalized.replace(source, target)
    return normalized
