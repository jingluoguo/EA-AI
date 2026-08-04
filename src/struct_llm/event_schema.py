from __future__ import annotations

from dataclasses import dataclass

from .structure import Frame, State


@dataclass(frozen=True)
class StateEffect:
    name: str
    left_role: str
    right_role: str


@dataclass(frozen=True)
class EventSchema:
    frame_type: str
    effects: tuple[StateEffect, ...] = ()
    qualifier_roles: tuple[tuple[str, str], ...] = ()

    def role_for_qualifier(self, key: str) -> str:
        aliases = dict(self.qualifier_roles)
        return aliases.get(key, key)


EVENT_SCHEMAS: dict[str, EventSchema] = {
    "put_in": EventSchema(
        "put_in",
        effects=(StateEffect("in", "theme", "goal"),),
        qualifier_roles=(("item", "theme"), ("holder", "goal")),
    ),
    "be_in": EventSchema(
        "be_in",
        effects=(StateEffect("in", "theme", "goal"),),
        qualifier_roles=(("item", "theme"), ("holder", "goal")),
    ),
    "take_out": EventSchema(
        "take_out",
        effects=(StateEffect("not_in", "theme", "source"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "not_in": EventSchema(
        "not_in",
        effects=(StateEffect("not_in", "theme", "source"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "move": EventSchema(
        "move",
        effects=(StateEffect("at", "theme", "goal"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "give": EventSchema(
        "give",
        effects=(StateEffect("owner", "theme", "recipient"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "paint": EventSchema(
        "paint",
        effects=(StateEffect("color", "theme", "result"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "open": EventSchema(
        "open",
        effects=(StateEffect("access", "theme", "result"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "close": EventSchema(
        "close",
        effects=(StateEffect("access", "theme", "result"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "create": EventSchema(
        "create",
        effects=(StateEffect("exists", "theme", "result"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "destroy": EventSchema(
        "destroy",
        effects=(StateEffect("exists", "theme", "result"),),
        qualifier_roles=(("item", "theme"),),
    ),
    "profile_name": EventSchema(
        "profile_name",
        effects=(StateEffect("name", "subject", "value"),),
    ),
    "profile_like": EventSchema(
        "profile_like",
        effects=(StateEffect("likes", "subject", "value"),),
    ),
    "profile_dislike": EventSchema(
        "profile_dislike",
        effects=(StateEffect("dislikes", "subject", "value"),),
    ),
}


def states_for_frame_schema(frame: Frame) -> tuple[State, ...]:
    schema = EVENT_SCHEMAS.get(frame.frame_type)
    if schema is None:
        return ()
    states: list[State] = []
    for effect in schema.effects:
        left = frame.role(effect.left_role)
        right = frame.role(effect.right_role)
        if left is None or right is None:
            continue
        states.append(State(effect.name, left, right, frame.frame_id))
    return tuple(states)


def frame_matches_qualifiers(
    frame: Frame,
    qualifiers: tuple[str, ...],
    ignored_keys: tuple[str, ...] = (),
) -> bool:
    schema = EVENT_SCHEMAS.get(frame.frame_type, EventSchema(frame.frame_type))
    for qualifier in qualifiers:
        if "=" not in qualifier:
            return False
        key, value = qualifier.split("=", 1)
        if key in ignored_keys:
            continue
        role_name = schema.role_for_qualifier(key)
        if frame.role(role_name) != value:
            return False
    return True
