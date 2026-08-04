from __future__ import annotations

import re

from .capabilities import StatementParser, StatementParseResult
from .normalization import normalize_container_slot, normalize_containment_expression, normalize_slot_value
from .structure import Entity, Frame, Role


PUT_IN_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)放进(?P<container>.+?)$")
PASSIVE_MOVE_RE = re.compile(r"^(?P<thing>.+?)被带到(?P<place>.+?)$")
ACTIVE_MOVE_RE = re.compile(r"^(?P<person>.+?)把(?P<thing>.+?)带到(?P<place>.+?)$")
GIVE_RE = re.compile(r"^(?P<giver>.+?)把(?P<item>.+?)交给(?P<receiver>.+?)$")
PAINT_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)涂成(?P<color>.+?)$")


def parse_statement(
    sentence: str,
    parsers: tuple[StatementParser, ...] | None = None,
) -> StatementParseResult | None:
    for parser in parsers or DEFAULT_STATEMENT_PARSERS:
        parsed = parser(sentence)
        if parsed is not None:
            return parsed
    return None


def parse_put_in_statement(sentence: str) -> StatementParseResult | None:
    sentence = normalize_containment_expression(sentence)
    put_in = PUT_IN_RE.match(sentence)
    if not put_in:
        return None
    data = put_in.groupdict()
    item = normalize_slot_value(data["item"])
    container = normalize_container_slot(data["container"])
    frame = frame_from_roles(
        "put_in",
        actor=data["person"],
        theme=item,
        goal=container,
    )
    return (
        [
            Entity("person", data["person"]),
            Entity("item", item),
            Entity("container", container),
        ],
        [frame, handle_frame(data["person"], item)],
    )


def parse_passive_move_statement(sentence: str) -> StatementParseResult | None:
    move = PASSIVE_MOVE_RE.match(sentence)
    if not move:
        return None
    data = move.groupdict()
    thing = normalize_slot_value(data["thing"])
    place = normalize_slot_value(data["place"])
    return (
        [
            Entity(moved_role(thing), thing),
            Entity("place", place),
        ],
        [frame_from_roles("move", theme=thing, goal=place)],
    )


def parse_active_move_statement(sentence: str) -> StatementParseResult | None:
    active_move = ACTIVE_MOVE_RE.match(sentence)
    if not active_move:
        return None
    data = active_move.groupdict()
    thing = normalize_slot_value(data["thing"])
    place = normalize_slot_value(data["place"])
    frame = frame_from_roles("move", actor=data["person"], theme=thing, goal=place)
    return (
        [
            Entity("person", data["person"]),
            Entity(moved_role(thing), thing),
            Entity("place", place),
        ],
        [frame, handle_frame(data["person"], thing)],
    )


def parse_give_statement(sentence: str) -> StatementParseResult | None:
    give = GIVE_RE.match(sentence)
    if not give:
        return None
    data = give.groupdict()
    frame = frame_from_roles(
        "give",
        actor=data["giver"],
        theme=data["item"],
        recipient=data["receiver"],
    )
    return (
        [
            Entity("giver", data["giver"]),
            Entity("receiver", data["receiver"]),
            Entity("item", data["item"]),
        ],
        [frame, handle_frame(data["giver"], data["item"])],
    )


def parse_paint_statement(sentence: str) -> StatementParseResult | None:
    paint = PAINT_RE.match(sentence)
    if not paint:
        return None
    data = paint.groupdict()
    frame = frame_from_roles(
        "paint",
        actor=data["person"],
        theme=data["item"],
        result=data["color"],
    )
    return (
        [
            Entity("person", data["person"]),
            Entity("item", data["item"]),
            Entity("color", data["color"]),
        ],
        [frame, handle_frame(data["person"], data["item"])],
    )


def frame_from_roles(frame_type: str, **roles: str) -> Frame:
    frame_id = "pending"
    return Frame(
        frame_id=frame_id,
        frame_type=frame_type,
        time=0,
        roles=tuple(Role(frame_id, name, value) for name, value in roles.items()),
    )


def with_time(frame: Frame, time: int) -> Frame:
    frame_id = f"f{time}"
    return Frame(
        frame_id=frame_id,
        frame_type=frame.frame_type,
        time=time,
        roles=tuple(Role(frame_id, role.name, role.value) for role in frame.roles),
    )


def handle_frame(actor: str, theme: str) -> Frame:
    return frame_from_roles("handle", actor=actor, theme=theme)


def moved_role(name: str) -> str:
    if name.endswith(("盒子", "背包", "抽屉", "托盘")):
        return "container"
    return "thing"


def dedupe_entities(entities: list[Entity]) -> tuple[Entity, ...]:
    by_name: dict[str, Entity] = {}
    for entity in entities:
        by_name.setdefault(entity.name, entity)
    return tuple(by_name.values())


DEFAULT_STATEMENT_PARSERS: tuple[StatementParser, ...] = (
    parse_put_in_statement,
    parse_passive_move_statement,
    parse_active_move_statement,
    parse_give_statement,
    parse_paint_statement,
)
