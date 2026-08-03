from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .structure import Entity, Event, Query, Relation, Structure


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Prediction:
    structure: Structure
    answer: str


PUT_IN_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)放进(?P<container>.+?)$")
MOVE_RE = re.compile(r"^(?P<thing>.+?)被带到(?P<place>.+?)$")
GIVE_RE = re.compile(r"^(?P<giver>.+?)把(?P<item>.+?)交给(?P<receiver>.+?)$")
PAINT_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)涂成(?P<color>.+?)$")

QUESTION_FRAMES = ("我想知道", "我想问", "想知道", "想问", "想了解", "帮我看看", "告诉我")
QUESTION_FILLERS = ("现在", "请问", "到底", "是", "的", "了", "吗", "呢")
PUT_IN_EVENT_QUESTION_RE = (
    re.compile(r"(?P<actor>谁)把(?P<item>.+?)放进(?P<holder>[^，,。？！?]+)"),
    re.compile(r"(?P<item>[^，,。？！?]+?)(?:是)?(?P<actor>谁)放进(?P<holder>[^，,。？！?]+)"),
    re.compile(r"(?P<item>[^，,。？！?]+?)被(?P<actor>谁)放进(?P<holder>[^，,。？！?]+)"),
)


def parse_text(text: str) -> Structure:
    entities: list[Entity] = []
    relations: list[Relation] = []
    events: list[Event] = []
    query_candidates: list[str] = []

    for sentence, is_question in _split_sentences(text):
        if is_question:
            query_candidates.append(sentence)
            continue

        extracted = _parse_statement(sentence)
        if extracted is None:
            query_candidates.append(sentence)
        if extracted is None:
            continue

        new_entities, new_relations, new_events = extracted
        entities.extend(new_entities)
        for relation in new_relations:
            _apply_state_relation(relations, relation)
        events.extend(new_events)

    if not entities and not relations and not events:
        raise ParseError(f"Cannot extract structure from text: {text}")

    deduped_entities = _dedupe_entities(entities)
    query = _parse_query_candidates(query_candidates, deduped_entities)
    structure = Structure(
        entities=deduped_entities,
        relations=tuple(dict.fromkeys(relations)),
        events=tuple(events),
        rules=(),
        query=query,
    )
    return Structure(
        entities=structure.entities,
        relations=structure.relations,
        events=structure.events,
        rules=_infer_rules(structure),
        query=structure.query,
    )


def answer_from_structure(structure: Structure) -> str:
    rule_set = set(structure.rules)

    if "event_actor_matches" in rule_set:
        query = _require_query(structure)
        actor = _actor_for_event_query(structure, query)
        item = _query_qualifier(query, "item")
        holder = _query_qualifier(query, "holder")
        if query.target == "put_in":
            return f"{actor}把{item}放进{holder}。"
        return f"{actor}执行了{query.target}。"

    if "actor_handles_item" in rule_set:
        query = _require_query(structure)
        event = _latest_event_for_target(structure, "handle", query.target)
        return f"{event.actor}拿的{query.target}。"

    if "holder_contains_things" in rule_set:
        query = _require_query(structure)
        contents = _contents_in_holder(structure, query.target)
        return f"{query.target}里至少有{_join_names(contents)}。"

    if "container_moves_contents" in rule_set:
        query = structure.query
        relation = _relation_for_left(structure, "in", query.target) if query else _only_relation(structure, "in")
        at_relation = _relation_for_left(structure, "at", relation.right)
        item = relation.left
        container = relation.right
        place = at_relation.right
        return f"{item}在{place}的{container}里。"

    if "transfer_changes_owner" in rule_set:
        query = structure.query
        owner = (
            _relation_for_left(structure, "owner", query.target)
            if query
            else _only_relation(structure, "owner")
        )
        item = owner.left
        receiver = owner.right
        return f"{receiver}拥有{item}。"

    if "paint_changes_color" in rule_set:
        query = structure.query
        color_relation = (
            _relation_for_left(structure, "color", query.target)
            if query
            else _only_relation(structure, "color")
        )
        item = color_relation.left
        color = color_relation.right
        return f"{item}是{color}。"

    raise ParseError(f"No rule can answer structure: {structure.linearize()}")


def predict(text: str) -> Prediction:
    structure = parse_text(text)
    return Prediction(structure=structure, answer=answer_from_structure(structure))


def _only_relation(structure: Structure, name: str) -> Relation:
    matches = [relation for relation in structure.relations if relation.name == name]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} relation, got {len(matches)}.")
    return matches[0]


def _only_event(structure: Structure, name: str) -> Event:
    matches = [event for event in structure.events if event.name == name]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} event, got {len(matches)}.")
    return matches[0]


def _split_sentences(text: str) -> tuple[tuple[str, bool], ...]:
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


def _parse_statement(sentence: str) -> tuple[list[Entity], list[Relation], list[Event]] | None:
    put_in = PUT_IN_RE.match(sentence)
    if put_in:
        data = put_in.groupdict()
        return (
            [
                Entity("person", data["person"]),
                Entity("item", data["item"]),
                Entity("container", data["container"]),
            ],
            [Relation("in", data["item"], data["container"])],
            [
                Event("put_in", data["person"], data["item"], (f"holder={data['container']}",)),
                Event("handle", data["person"], data["item"]),
            ],
        )

    move = MOVE_RE.match(sentence)
    if move:
        data = move.groupdict()
        return (
            [
                Entity(_moved_role(data["thing"]), data["thing"]),
                Entity("place", data["place"]),
            ],
            [Relation("at", data["thing"], data["place"])],
            [Event("move", data["thing"], data["place"])],
        )

    give = GIVE_RE.match(sentence)
    if give:
        data = give.groupdict()
        return (
            [
                Entity("giver", data["giver"]),
                Entity("receiver", data["receiver"]),
                Entity("item", data["item"]),
            ],
            [Relation("owner", data["item"], data["receiver"])],
            [
                Event("give", data["giver"], data["receiver"]),
                Event("handle", data["giver"], data["item"]),
            ],
        )

    paint = PAINT_RE.match(sentence)
    if paint:
        data = paint.groupdict()
        return (
            [
                Entity("person", data["person"]),
                Entity("item", data["item"]),
                Entity("color", data["color"]),
            ],
            [Relation("color", data["item"], data["color"])],
            [
                Event("paint", data["item"], data["color"]),
                Event("handle", data["person"], data["item"]),
            ],
        )

    return None


def _parse_query_candidates(candidates: list[str], entities: tuple[Entity, ...]) -> Optional[Query]:
    if not candidates:
        return None

    errors: list[ParseError] = []
    for candidate in reversed(candidates):
        try:
            return _parse_query(candidate, entities)
        except ParseError as error:
            errors.append(error)

    combined = "，".join(candidates)
    try:
        return _parse_query(combined, entities)
    except ParseError:
        raise errors[-1]


def _parse_query(sentence: str, entities: tuple[Entity, ...]) -> Query:
    normalized = _normalize_question(sentence)
    event_query = _parse_event_query(normalized, entities)
    if event_query is not None:
        return event_query

    if "哪里" in normalized or "哪儿" in normalized:
        target = _extract_query_target(normalized, ("哪里", "哪儿", "在"), entities)
        return Query("location", target)

    if "谁" in normalized and "拥有" in normalized:
        target = _extract_query_target(normalized, ("谁", "拥有"), entities)
        return Query("owner", target)

    if "颜色" in normalized and "什么" in normalized:
        target = _extract_query_target(normalized, ("什么", "颜色"), entities)
        return Query("color", target)

    if "谁" in normalized and "拿" in normalized:
        target = _extract_query_target(normalized, ("谁", "拿"), entities)
        return Query("actor_for_item", target)

    if "什么" in normalized and "有" in normalized:
        target = _extract_query_target(normalized, ("什么", "有", "里", "至少"), entities)
        return Query("contents", target)

    raise ParseError(f"Cannot parse question: {sentence}")


def _parse_event_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    put_in = _parse_put_in_event_question(sentence, entities)
    if put_in is not None:
        return put_in

    return None


def _parse_put_in_event_question(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    for pattern in PUT_IN_EVENT_QUESTION_RE:
        match = pattern.search(sentence)
        if match and match.group("actor") == "谁":
            return _event_actor_query("put_in", match.group("item"), match.group("holder"), entities)
    return None


def _event_actor_query(event_name: str, item: str, holder: str, entities: tuple[Entity, ...]) -> Query:
    return Query(
        "actor_for_event",
        event_name,
        (
            f"item={_normalize_entity_slot(item, entities)}",
            f"holder={_normalize_entity_slot(holder, entities)}",
        ),
    )


def _normalize_question(sentence: str) -> str:
    normalized = sentence.strip().replace("？", "").replace("?", "")
    changed = True
    while changed:
        previous = normalized
        normalized = _strip_question_frames(_normalize_slot_value(normalized))
        changed = normalized != previous
    return normalized


def _strip_question_frames(sentence: str) -> str:
    normalized = sentence.strip()
    changed = True
    while changed:
        changed = False
        for frame in sorted(QUESTION_FRAMES, key=len, reverse=True):
            if normalized.startswith(frame) and len(normalized) > len(frame):
                normalized = normalized[len(frame) :]
                changed = True
    return normalized


def _normalize_slot_value(value: str) -> str:
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


def _extract_query_target(
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
    return _normalize_entity_slot(target, entities)


def _normalize_entity_slot(value: str, entities: tuple[Entity, ...]) -> str:
    normalized = _normalize_slot_value(value)
    matches = [entity.name for entity in entities if entity.name in normalized]
    if matches:
        return max(matches, key=len)
    return normalized


def _infer_rules(structure: Structure) -> tuple[str, ...]:
    if structure.query is None:
        return ()

    rules: list[str] = []
    query = structure.query

    if query.intent == "actor_for_event" and _has_actor_for_event_query(structure, query):
        rules.append("event_actor_matches")

    if query.intent == "actor_for_item" and _has_event_target(structure, "handle", query.target):
        rules.append("actor_handles_item")

    if query.intent == "contents" and _contents_in_holder(structure, query.target):
        rules.append("holder_contains_things")

    if query.intent == "location":
        in_relations = [relation for relation in structure.relations if relation.name == "in" and relation.left == query.target]
        if any(_has_relation_left(structure, "at", relation.right) for relation in in_relations):
            rules.append("container_moves_contents")

    if query.intent == "owner" and _has_relation_left(structure, "owner", query.target):
        rules.append("transfer_changes_owner")

    if query.intent == "color" and _has_relation_left(structure, "color", query.target):
        rules.append("paint_changes_color")

    return tuple(rules)


def _dedupe_entities(entities: list[Entity]) -> tuple[Entity, ...]:
    by_name: dict[str, Entity] = {}
    for entity in entities:
        by_name.setdefault(entity.name, entity)
    return tuple(by_name.values())


def _apply_state_relation(relations: list[Relation], relation: Relation) -> None:
    if relation.name in {"in", "at", "owner", "color"}:
        relations[:] = [
            existing
            for existing in relations
            if not (existing.name == relation.name and existing.left == relation.left)
        ]
    relations.append(relation)


def _moved_role(name: str) -> str:
    if name.endswith(("盒子", "背包", "抽屉", "托盘")):
        return "container"
    return "thing"


def _contents_in_holder(structure: Structure, holder: str) -> tuple[str, ...]:
    contents: list[str] = []
    frontier = _direct_contents(structure, holder)

    while frontier:
        current = frontier.pop(0)
        if current in contents:
            continue
        contents.append(current)
        frontier.extend(item for item in _direct_contents(structure, current) if item not in contents)

    return tuple(contents)


def _direct_contents(structure: Structure, holder: str) -> list[str]:
    contents = [relation.left for relation in structure.relations if relation.name == "at" and relation.right == holder]
    contents.extend(
        relation.left for relation in structure.relations if relation.name == "in" and relation.right == holder
    )
    return contents


def _join_names(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return "和".join(names)


def _has_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        _actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True


def _actor_for_event_query(structure: Structure, query: Query) -> str:
    item = _query_qualifier(query, "item")
    holder = _query_qualifier(query, "holder")
    matches = [
        event.actor
        for event in structure.events
        if event.name == query.target
        and event.target == item
        and _event_qualifier(event, "holder") == holder
    ]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one actor for event query, got {len(matches)}.")
    return matches[0]


def _query_qualifier(query: Query, key: str) -> str:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in query.qualifiers if qualifier.startswith(prefix)]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one query qualifier {key}, got {len(matches)}.")
    return matches[0]


def _event_qualifier(event: Event, key: str) -> str | None:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in event.qualifiers if qualifier.startswith(prefix)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ParseError(f"Expected at most one event qualifier {key}, got {len(matches)}.")
    return matches[0]


def _require_query(structure: Structure) -> Query:
    if structure.query is None:
        raise ParseError("Expected query in structure.")
    return structure.query


def _relation_for_left(structure: Structure, name: str, left: str) -> Relation:
    matches = [relation for relation in structure.relations if relation.name == name and relation.left == left]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} relation for {left}, got {len(matches)}.")
    return matches[0]


def _relation_for_right(structure: Structure, name: str, right: str) -> Relation:
    matches = [relation for relation in structure.relations if relation.name == name and relation.right == right]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} relation for {right}, got {len(matches)}.")
    return matches[0]


def _event_for_actor(structure: Structure, name: str, actor: str) -> Event:
    matches = [event for event in structure.events if event.name == name and event.actor == actor]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} event for {actor}, got {len(matches)}.")
    return matches[0]


def _event_for_target(structure: Structure, name: str, target: str) -> Event:
    matches = [event for event in structure.events if event.name == name and event.target == target]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} event for {target}, got {len(matches)}.")
    return matches[0]


def _latest_event_for_target(structure: Structure, name: str, target: str) -> Event:
    matches = [event for event in structure.events if event.name == name and event.target == target]
    if not matches:
        raise ParseError(f"Expected at least one {name} event for {target}.")
    return matches[-1]


def _has_relation_right(structure: Structure, name: str, right: str) -> bool:
    return any(relation.name == name and relation.right == right for relation in structure.relations)


def _has_relation_left(structure: Structure, name: str, left: str) -> bool:
    return any(relation.name == name and relation.left == left for relation in structure.relations)


def _has_event(structure: Structure, name: str) -> bool:
    return any(event.name == name for event in structure.events)


def _has_event_actor(structure: Structure, name: str, actor: str) -> bool:
    return any(event.name == name and event.actor == actor for event in structure.events)


def _has_event_target(structure: Structure, name: str, target: str) -> bool:
    return any(event.name == name and event.target == target for event in structure.events)
