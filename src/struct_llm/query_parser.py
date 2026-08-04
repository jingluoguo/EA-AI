from __future__ import annotations

import re
from typing import Optional

from .capabilities import QueryParser
from .errors import ParseError
from .normalization import (
    QUESTION_FILLERS,
    normalize_container_slot,
    normalize_entity_slot,
    normalize_question,
)
from .structure import Entity, Query


PUT_IN_EVENT_QUESTION_RE = (
    re.compile(r"(?P<actor>谁)把(?P<item>.+?)放进(?P<holder>[^，,。？！?]+)"),
    re.compile(r"(?P<item>[^，,。？！?]+?)(?:是)?(?P<actor>谁)放进(?P<holder>[^，,。？！?]+)"),
    re.compile(r"(?P<item>[^，,。？！?]+?)被(?P<actor>谁)放进(?P<holder>[^，,。？！?]+)"),
)


def parse_query_candidates(
    candidates: list[str],
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...] | None = None,
) -> Optional[Query]:
    if not candidates:
        return None

    errors: list[ParseError] = []
    for candidate in reversed(candidates):
        try:
            return parse_query(candidate, entities, parsers)
        except ParseError as error:
            errors.append(error)

    combined = "，".join(candidates)
    try:
        return parse_query(combined, entities, parsers)
    except ParseError:
        raise errors[-1]


def parse_query(
    sentence: str,
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...] | None = None,
) -> Query:
    normalized = normalize_question(sentence)
    for parser in parsers or DEFAULT_QUERY_PARSERS:
        query = parser(normalized, entities)
        if query is not None:
            return query

    raise ParseError(f"Cannot parse question: {sentence}")


def parse_event_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    put_in = parse_put_in_event_question(sentence, entities)
    if put_in is not None:
        return put_in

    return None


def parse_location_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "哪里" in sentence or "哪儿" in sentence:
        target = extract_query_target(sentence, ("哪里", "哪儿", "在"), entities)
        return Query("location", target)
    return None


def parse_owner_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "谁" in sentence and "拥有" in sentence:
        target = extract_query_target(sentence, ("谁", "拥有"), entities)
        return Query("owner", target)
    return None


def parse_color_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "颜色" in sentence and "什么" in sentence:
        target = extract_query_target(sentence, ("什么", "颜色"), entities)
        return Query("color", target)
    return None


def parse_actor_for_item_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "谁" in sentence and "拿" in sentence:
        target = extract_query_target(sentence, ("谁", "拿"), entities)
        return Query("actor_for_item", target)
    return None


def parse_contents_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "什么" in sentence and "有" in sentence:
        target = extract_query_target(sentence, ("什么", "有", "里", "至少"), entities)
        return Query("contents", target)
    return None


DEFAULT_QUERY_PARSERS: tuple[QueryParser, ...] = (
    parse_event_query,
    parse_location_query,
    parse_owner_query,
    parse_color_query,
    parse_actor_for_item_query,
    parse_contents_query,
)


def parse_put_in_event_question(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    for pattern in PUT_IN_EVENT_QUESTION_RE:
        match = pattern.search(sentence)
        if match and match.group("actor") == "谁":
            return event_actor_query("put_in", match.group("item"), match.group("holder"), entities)
    return None


def event_actor_query(event_name: str, item: str, holder: str, entities: tuple[Entity, ...]) -> Query:
    return Query(
        "actor_for_event",
        event_name,
        (
            f"item={normalize_entity_slot(item, entities)}",
            f"holder={normalize_entity_slot(normalize_container_slot(holder), entities)}",
        ),
    )


def extract_query_target(
    sentence: str,
    intent_words: tuple[str, ...],
    entities: tuple[Entity, ...],
) -> str:
    target = sentence
    words = sorted(set((*intent_words, *QUESTION_FILLERS)), key=len, reverse=True)
    for word in words:
        target = target.replace(word, "")
    if not target:
        raise ParseError(f"Cannot extract query target from question: {sentence}")
    return normalize_entity_slot(target, entities)
