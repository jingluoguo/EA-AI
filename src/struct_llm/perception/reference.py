from __future__ import annotations

from ..structure import Entity


OBJECT_PRONOUNS = ("这个东西", "这件东西", "这个物品", "它")
PERSON_PRONOUNS = ("这个人", "他", "她")
PLACE_PRONOUNS = ("这个地方", "这里", "那儿", "那里")
RELATIVE_PRONOUNS = ("前者", "后者")


def resolve_references(sentence: str, entities: tuple[Entity, ...]) -> str:
    resolved = sentence
    resolved = resolve_typed_demonstratives(resolved, entities)
    resolved = resolve_relative_pronouns(resolved, entities)
    resolved = replace_first_available(resolved, PLACE_PRONOUNS, latest_entity(entities, ("place",)))
    resolved = replace_first_available(
        resolved,
        PERSON_PRONOUNS,
        latest_entity(entities, ("person", "giver", "receiver")),
    )
    resolved = replace_first_available(
        resolved,
        OBJECT_PRONOUNS,
        latest_entity(entities, ("item", "container", "thing")),
    )
    return resolved


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
        for prefix in ("这个", "这件", "那个", "那件"):
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


def last_two_salient_entities(entities: tuple[Entity, ...]) -> tuple[str | None, str | None]:
    salient_roles = ("item", "container", "thing", "person", "giver", "receiver")
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
