from __future__ import annotations

from ...structure import Query, State, Structure
from ...world.state import apply_state, states_from_frame
from .common import object_exists_value, query_qualifier, structure_with_states
from .event import temporal_anchor_frame
from .location import location_path
from .common import states_before_time

__all__ = (
    "polar_contents_status",
    "contents_in_holder",
    "direct_contents",
    "inventory_by_owner",
    "places_visited",
    "temporal_event_contents",
)


def polar_contents_status(structure: Structure, query: Query) -> bool | None:
    item = query_qualifier(query, "item")
    if object_exists_value(structure, item) == "不存在":
        return False
    contents = contents_in_holder(structure, query.target)
    if item in contents:
        return True
    if any(
        state.name == "not_in" and state.left == item and state.right == query.target
        for state in structure.states
    ):
        return False
    if contents:
        return False
    return None


def contents_in_holder(structure: Structure, holder: str) -> tuple[str, ...]:
    contents: list[str] = []
    frontier = direct_contents(structure, holder)

    while frontier:
        current = frontier.pop(0)
        if current in contents:
            continue
        contents.append(current)
        frontier.extend(item for item in direct_contents(structure, current) if item not in contents)

    return tuple(contents)


def direct_contents(structure: Structure, holder: str) -> list[str]:
    contents = [state.left for state in structure.states if state.name == "at" and state.right == holder]
    contents.extend(state.left for state in structure.states if state.name == "in" and state.right == holder)
    return [content for content in contents if object_exists_value(structure, content) != "不存在"]


def inventory_by_owner(structure: Structure) -> dict[str, tuple[str, ...]]:
    inventories: dict[str, list[str]] = {}
    for state in structure.states:
        if state.name == "owner":
            inventories.setdefault(state.right, []).append(state.left)
    return {owner: tuple(items) for owner, items in inventories.items()}


def places_visited(structure: Structure, target: str) -> tuple[str, ...]:
    states: list[State] = []
    places: list[str] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        for state in states_from_frame(frame):
            apply_state(states, state)
        snapshot = structure_with_states(structure, states)
        place, _ = location_path(snapshot, target)
        if place is not None and place not in places:
            places.append(place)
    return tuple(places)


def temporal_event_contents(
    structure: Structure,
    query: Query,
    include_anchor: bool,
) -> tuple[str, ...] | None:
    anchor = temporal_anchor_frame(structure, query)
    if anchor is None:
        return None
    states = states_before_time(structure, anchor.time + 1) if include_anchor else states_before_time(structure, anchor.time)
    snapshot = structure_with_states(structure, states)
    contents = contents_in_holder(snapshot, query.target)
    if not contents:
        return None
    return contents
