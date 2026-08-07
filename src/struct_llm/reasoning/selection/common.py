from __future__ import annotations

from ...errors import ParseError
from ...structure import Frame, Query, State, Structure
from ...world.state import apply_state, states_from_frame

__all__ = (
    "query_qualifier",
    "optional_query_qualifier",
    "split_qualifier",
    "require_query",
    "only_state",
    "state_for_left",
    "state_for_left_or_none",
    "has_state_left",
    "object_exists_value",
    "object_is_known",
    "has_frame_with_role",
    "latest_frame_with_role",
    "required_frame_role",
    "join_names",
    "structure_with_states",
    "structure_with_query",
    "states_before_time",
    "profile_values",
)

def query_qualifier(query: Query, key: str) -> str:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in query.qualifiers if qualifier.startswith(prefix)]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one query qualifier {key}, got {len(matches)}.")
    return matches[0]
def optional_query_qualifier(query: Query, key: str) -> str | None:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in query.qualifiers if qualifier.startswith(prefix)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ParseError(f"Expected at most one query qualifier {key}, got {len(matches)}.")
    return matches[0]
def split_qualifier(qualifier: str) -> tuple[str, str]:
    if "=" not in qualifier:
        raise ParseError(f"Expected query qualifier key=value, got {qualifier}.")
    key, value = qualifier.split("=", 1)
    return key, value
def require_query(structure: Structure) -> Query:
    if structure.query is None:
        raise ParseError("Expected query in structure.")
    return structure.query
def only_state(structure: Structure, name: str) -> State:
    matches = [state for state in structure.states if state.name == name]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} state, got {len(matches)}.")
    return matches[0]
def state_for_left(structure: Structure, name: str, left: str) -> State:
    matches = [state for state in structure.states if state.name == name and state.left == left]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} state for {left}, got {len(matches)}.")
    return matches[0]
def state_for_left_or_none(structure: Structure, name: str, left: str) -> State | None:
    matches = [state for state in structure.states if state.name == name and state.left == left]
    return matches[-1] if matches else None
def has_state_left(structure: Structure, name: str, left: str) -> bool:
    return any(state.name == name and state.left == left for state in structure.states)
def object_exists_value(structure: Structure, target: str) -> str | None:
    states = [state for state in structure.states if state.name == "exists" and state.left == target]
    return states[-1].right if states else None
def object_is_known(structure: Structure, target: str) -> bool:
    if any(entity.name == target for entity in structure.entities):
        return True
    if any(state.left == target or state.right == target for state in structure.states):
        return True
    if any(role.value == target for frame in structure.frames for role in frame.roles):
        return True
    return False
def has_frame_with_role(structure: Structure, frame_type: str, role_name: str, value: str) -> bool:
    return any(frame.frame_type == frame_type and frame.role(role_name) == value for frame in structure.frames)
def latest_frame_with_role(structure: Structure, frame_type: str, role_name: str, value: str) -> Frame:
    matches = [frame for frame in structure.frames if frame.frame_type == frame_type and frame.role(role_name) == value]
    if not matches:
        raise ParseError(f"Expected at least one {frame_type} frame with {role_name}={value}.")
    return matches[-1]
def required_frame_role(frame: Frame, role_name: str) -> str:
    value = frame.role(role_name)
    if value is None:
        raise ParseError(f"Expected role {role_name} in frame {frame.frame_id}.")
    return value
def join_names(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return "和".join(names)
def structure_with_states(structure: Structure, states: list[State]) -> Structure:
    return Structure(
        entities=structure.entities,
        rules=structure.rules,
        relations=tuple(state.to_relation() for state in states),
        events=structure.events,
        query=structure.query,
        frames=structure.frames,
        states=tuple(states),
        intentions=structure.intentions,
    )
def structure_with_query(structure: Structure, query: Query) -> Structure:
    return Structure(
        entities=structure.entities,
        rules=(),
        relations=structure.relations,
        events=structure.events,
        query=query,
        frames=structure.frames,
        states=structure.states,
        intentions=structure.intentions,
    )
def states_before_time(structure: Structure, time: int) -> list[State]:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        if frame.time >= time:
            break
        for state in states_from_frame(frame):
            apply_state(states, state)
    return states
def profile_values(structure: Structure, target: str, attribute: str) -> tuple[str, ...]:
    return tuple(
        state.right
        for state in structure.states
        if state.name == attribute and state.left == target
    )
