from __future__ import annotations

import re
from dataclasses import dataclass

from .structure import Entity, Event, Relation, Structure


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Prediction:
    structure: Structure
    answer: str


CONTAINMENT_RE = re.compile(
    r"^(?P<person>.+?)把(?P<item>.+?)放进(?P<container>.+?)。"
    r"(?P=container)被带到(?P<place>.+?)。"
    r"(?P=item)在哪里？$"
)

OWNERSHIP_RE = re.compile(
    r"^(?P<giver>.+?)把(?P<item>.+?)交给(?P<receiver>.+?)。"
    r"现在谁拥有(?P=item)？$"
)

COLOR_RE = re.compile(
    r"^(?P<person>.+?)把(?P<item>.+?)涂成(?P<color>.+?)。"
    r"现在(?P=item)是什么颜色？$"
)


def parse_text(text: str) -> Structure:
    containment = CONTAINMENT_RE.match(text)
    if containment:
        data = containment.groupdict()
        return Structure(
            entities=(
                Entity("person", data["person"]),
                Entity("item", data["item"]),
                Entity("container", data["container"]),
                Entity("place", data["place"]),
            ),
            relations=(Relation("in", data["item"], data["container"]),),
            events=(Event("move", data["container"], data["place"]),),
            rules=("container_moves_contents",),
        )

    ownership = OWNERSHIP_RE.match(text)
    if ownership:
        data = ownership.groupdict()
        return Structure(
            entities=(
                Entity("giver", data["giver"]),
                Entity("receiver", data["receiver"]),
                Entity("item", data["item"]),
            ),
            relations=(Relation("owns_before", data["giver"], data["item"]),),
            events=(Event("give", data["giver"], data["receiver"]),),
            rules=("transfer_changes_owner",),
        )

    color = COLOR_RE.match(text)
    if color:
        data = color.groupdict()
        return Structure(
            entities=(
                Entity("person", data["person"]),
                Entity("item", data["item"]),
                Entity("color", data["color"]),
            ),
            relations=(),
            events=(Event("paint", data["item"], data["color"]),),
            rules=("paint_changes_color",),
        )

    raise ParseError(f"Cannot parse text: {text}")


def answer_from_structure(structure: Structure) -> str:
    rule_set = set(structure.rules)

    if "container_moves_contents" in rule_set:
        relation = _only_relation(structure, "in")
        event = _only_event(structure, "move")
        item = relation.left
        container = relation.right
        place = event.target
        if event.actor != container:
            raise ParseError("Move event actor must be the same container that holds the item.")
        return f"{item}在{place}的{container}里。"

    if "transfer_changes_owner" in rule_set:
        owns_before = _only_relation(structure, "owns_before")
        event = _only_event(structure, "give")
        item = owns_before.right
        receiver = event.target
        return f"{receiver}拥有{item}。"

    if "paint_changes_color" in rule_set:
        event = _only_event(structure, "paint")
        item = event.actor
        color = event.target
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
