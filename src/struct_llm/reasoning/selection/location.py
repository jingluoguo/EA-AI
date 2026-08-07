from __future__ import annotations

from ...errors import ParseError
from ...structure import Query, State, Structure
from ...world.state import apply_state, states_from_frame
from .common import object_exists_value, query_qualifier, state_for_left, states_before_time, structure_with_states
from .event import first_actor_action_frame, frame_matches_counterfactual_exclusion, temporal_anchor_frame

__all__ = (
    "polar_location_status",
    "same_location_status",
    "location_key",
    "location_path",
    "container_chain_text",
    "location_phrase",
    "describe_object_location",
    "first_location",
    "location_before_actor_action",
    "temporal_event_location",
    "counterfactual_location",
)

def polar_location_status(structure: Structure, query: Query) -> bool | None:
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    if object_exists_value(structure, query.target) == "不存在":
        return False
    place, containers = location_path(structure, query.target)
    if kind == "at":
        if place is None and not containers:
            return None
        if place is None:
            return False
        return place == expected
    if place is None and not containers:
        return None
    return expected in containers
def same_location_status(structure: Structure, query: Query) -> bool | None:
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    left_key = location_key(structure, left)
    right_key = location_key(structure, right)
    if left_key is None or right_key is None:
        return None
    return left_key == right_key
def location_key(structure: Structure, target: str) -> tuple[str, str] | None:
    if object_exists_value(structure, target) == "不存在":
        return None
    place, containers = location_path(structure, target)
    if place is not None:
        return ("place", place)
    if containers:
        return ("container", containers[-1])
    return None
def location_path(structure: Structure, target: str) -> tuple[str | None, tuple[str, ...]]:
    if object_exists_value(structure, target) == "不存在":
        return None, ()

    containers: list[str] = []
    current = target
    visited = {target}

    while True:
        at_states = [state for state in structure.states if state.name == "at" and state.left == current]
        if at_states:
            return at_states[-1].right, tuple(containers)

        in_states = [state for state in structure.states if state.name == "in" and state.left == current]
        if not in_states:
            return None, tuple(containers)

        container = in_states[-1].right
        if container in visited:
            raise ParseError(f"Containment cycle detected at {container}.")
        containers.append(container)
        visited.add(container)
        current = container
def container_chain_text(containers: tuple[str, ...]) -> str:
    if not containers:
        raise ParseError("Expected at least one container.")
    ordered = tuple(reversed(containers))
    text = ordered[0]
    for container in ordered[1:]:
        text = f"{text}里的{container}"
    return f"{text}里"
def location_phrase(place: str | None, containers: tuple[str, ...]) -> str:
    if place is not None and containers:
        return f"在{place}的{container_chain_text(containers)}"
    if place is not None:
        return f"在{place}"
    if containers:
        return f"在{container_chain_text(containers)}"
    raise ParseError("Expected a place or container path.")
def describe_object_location(structure: Structure, target: str) -> str:
    if object_exists_value(structure, target) == "不存在":
        return "不存在"
    place, containers = location_path(structure, target)
    if place is None and not containers:
        return "在哪里未知"
    return location_phrase(place, containers)
def first_location(structure: Structure, target: str) -> tuple[str | None, tuple[str, ...]] | None:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        for state in states_from_frame(frame):
            apply_state(states, state)
        snapshot = structure_with_states(structure, states)
        place, containers = location_path(snapshot, target)
        if place is not None or containers:
            return place, containers
    return None
def location_before_actor_action(
    structure: Structure,
    target: str,
    actor: str,
) -> tuple[str | None, tuple[str, ...]] | None:
    action = first_actor_action_frame(structure, target, actor)
    if action is None:
        return None
    states = states_before_time(structure, action.time)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, target)
    if place is None and not containers:
        return None
    return place, containers
def temporal_event_location(
    structure: Structure,
    query: Query,
    include_anchor: bool,
) -> tuple[str | None, tuple[str, ...]] | None:
    anchor = temporal_anchor_frame(structure, query)
    if anchor is None:
        return None
    states = states_before_time(structure, anchor.time + 1) if include_anchor else states_before_time(structure, anchor.time)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, query.target)
    if place is None and not containers:
        return None
    return place, containers
def counterfactual_location(structure: Structure, query: Query) -> tuple[str | None, tuple[str, ...]] | None:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        if frame_matches_counterfactual_exclusion(frame, query):
            continue
        for state in states_from_frame(frame):
            apply_state(states, state)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, query.target)
    if place is None and not containers:
        return None
    return place, containers
