from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..dataset_io import load_jsonl_objects
from ..structure import Frame, State


EVENT_SCHEMA_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "event_schema_examples.jsonl"
EVENT_SCHEMA_RECORD_SCHEMA = "struct_llm.event_schema.v1"


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


def load_event_schema_jsonl(path: str | Path) -> tuple[EventSchema, ...]:
    return tuple(
        event_schema_from_dict(record, line_number=line_number)
        for line_number, record in enumerate(load_jsonl_objects(path, "event schema"), start=1)
    )


@lru_cache(maxsize=None)
def event_schemas(path: str | Path = EVENT_SCHEMA_DATA_PATH) -> dict[str, EventSchema]:
    return {schema.frame_type: schema for schema in load_event_schema_jsonl(path)}


def event_schema_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> EventSchema:
    prefix = f"Event schema at line {line_number}" if line_number is not None else "Event schema"
    schema = str(record.get("schema") or EVENT_SCHEMA_RECORD_SCHEMA).strip()
    if schema != EVENT_SCHEMA_RECORD_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema: {schema}")
    frame_type = str(record.get("frame_type") or "").strip()
    if not frame_type:
        raise ValueError(f"{prefix} requires frame_type.")
    raw_effects = record.get("effects", ())
    if not isinstance(raw_effects, list):
        raise ValueError(f"{prefix} effects must be a list.")
    raw_qualifier_roles = record.get("qualifier_roles", {})
    if not isinstance(raw_qualifier_roles, dict):
        raise ValueError(f"{prefix} qualifier_roles must be an object.")
    return EventSchema(
        frame_type=frame_type,
        effects=tuple(state_effect_from_dict(value, prefix) for value in raw_effects),
        qualifier_roles=tuple(
            (str(key).strip(), str(value).strip())
            for key, value in raw_qualifier_roles.items()
            if str(key).strip() and str(value).strip()
        ),
    )


def state_effect_from_dict(record: Any, prefix: str) -> StateEffect:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} effect entries must be objects.")
    name = str(record.get("name") or "").strip()
    left_role = str(record.get("left_role") or "").strip()
    right_role = str(record.get("right_role") or "").strip()
    if not name or not left_role or not right_role:
        raise ValueError(f"{prefix} effects require name, left_role and right_role.")
    return StateEffect(name, left_role, right_role)


EVENT_SCHEMAS = event_schemas()


def states_for_frame_schema(frame: Frame) -> tuple[State, ...]:
    schema = event_schemas().get(frame.frame_type)
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
    schema = event_schemas().get(frame.frame_type, EventSchema(frame.frame_type))
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
