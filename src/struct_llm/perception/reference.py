from __future__ import annotations

import unicodedata

from ..comprehension.surface_lexicon import surface_forms
from ..structure import Entity


def resolve_references(sentence: str, entities: tuple[Entity, ...]) -> str:
    resolved = sentence
    resolved = resolve_typed_demonstratives(resolved, entities)
    resolved = resolve_relative_pronouns(resolved, entities)
    resolved = replace_first_available(resolved, surface_forms("place_pronoun"), latest_entity(entities, ("place",)))
    resolved = replace_first_available(
        resolved,
        surface_forms("person_pronoun"),
        latest_entity(entities, ("person", "giver", "receiver")),
    )
    resolved = replace_first_available(
        resolved,
        surface_forms("object_pronoun"),
        latest_entity(entities, ("item", "container", "thing")) or unique_object_focus_topic(entities),
    )
    resolved = resolve_bare_demonstratives(resolved, entities)
    return resolved


def unresolved_reference_pronoun(sentence: str) -> str | None:
    """Return a reference form that remains for discourse resolution."""
    for pronoun in sorted(
        (
            *surface_forms("object_pronoun"),
            *surface_forms("person_pronoun"),
            *surface_forms("place_pronoun"),
            *surface_forms("bare_demonstrative"),
        ),
        key=len,
        reverse=True,
    ):
        if pronoun in sentence:
            return pronoun
    return None


def unresolved_reference_mention(sentence: str, resolved_sentence: str) -> Entity | None:
    """Expose an unresolved reference as a perception-layer entity."""
    pronoun = unresolved_reference_pronoun(sentence)
    if pronoun is None or pronoun not in resolved_sentence:
        return None
    return Entity("unresolved_reference", pronoun)


def resolve_relative_pronouns(sentence: str, entities: tuple[Entity, ...]) -> str:
    first, second = last_two_salient_entities(entities)
    if first is None or second is None:
        return sentence
    resolved = sentence.replace("前者", first)
    resolved = resolved.replace("后者", second)
    return resolved


def resolve_typed_demonstratives(sentence: str, entities: tuple[Entity, ...]) -> str:
    resolved = sentence
    for entity in sorted(entities, key=lambda candidate: len(candidate.name), reverse=True):
        for prefix in surface_forms("demonstrative_prefix"):
            phrase = f"{prefix}{entity.name}"
            if phrase in resolved:
                resolved = resolved.replace(phrase, entity.name)
    return resolved


def replace_first_available(sentence: str, pronouns: tuple[str, ...], replacement: str | None) -> str:
    if replacement is None:
        return sentence
    resolved = sentence
    for pronoun in sorted(pronouns, key=len, reverse=True):
        resolved = resolved.replace(pronoun, replacement)
    return resolved


def latest_entity(entities: tuple[Entity, ...], roles: tuple[str, ...]) -> str | None:
    for entity in reversed(entities):
        if entity.role in roles:
            return entity.name
    return None


def resolve_bare_demonstratives(sentence: str, entities: tuple[Entity, ...]) -> str:
    stripped = sentence.strip()
    replacement = unique_object_focus_topic(entities) or unique_salient_entity(entities)
    if replacement is None:
        return sentence
    for demonstrative in surface_forms("bare_demonstrative"):
        if strip_ellipsis_particles(stripped) == demonstrative:
            return sentence.replace(demonstrative, replacement, 1)
    return sentence


def strip_ellipsis_particles(sentence: str) -> str:
    stripped = strip_terminal_punctuation(sentence)
    particles = surface_forms("terminal_discourse_particle")
    while any(stripped.endswith(particle) for particle in particles):
        particle = next(particle for particle in particles if stripped.endswith(particle))
        stripped = stripped[: -len(particle)].strip()
        stripped = strip_terminal_punctuation(stripped)
    return stripped


def strip_terminal_punctuation(sentence: str) -> str:
    stripped = sentence.strip()
    while stripped and unicodedata.category(stripped[-1]).startswith("P"):
        stripped = stripped[:-1].strip()
    return stripped


def unique_object_focus_topic(entities: tuple[Entity, ...]) -> str | None:
    query_intents = {entity.name for entity in entities if entity.role == "query_intent" and entity.name}
    if not (query_intents & set(surface_forms("object_followup_query_intent"))):
        return None
    topics = tuple(dict.fromkeys(entity.name for entity in entities if entity.role == "topic" and entity.name))
    if len(topics) != 1:
        return None
    return topics[0]


def unique_salient_entity(entities: tuple[Entity, ...]) -> str | None:
    salient_roles = surface_forms("salient_reference_role")
    names = tuple(dict.fromkeys(entity.name for entity in entities if entity.role in salient_roles and entity.name))
    if len(names) != 1:
        return None
    return names[0]


def last_two_salient_entities(entities: tuple[Entity, ...]) -> tuple[str | None, str | None]:
    salient_roles = surface_forms("salient_reference_role")
    seen: list[str] = []
    for entity in entities:
        if entity.role not in salient_roles:
            continue
        if entity.name in seen:
            continue
        seen.append(entity.name)
    if len(seen) < 2:
        return None, None
    return seen[-2], seen[-1]
