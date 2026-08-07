from __future__ import annotations

from ...structure import Structure
from ..selectors import *

__all__ = (
    "infer_initial_location",
    "infer_location_before_actor_action",
    "infer_location_before_event",
    "infer_location_after_event",
    "infer_counterfactual_location",
    "infer_polar_location",
    "infer_same_location",
    "infer_places_visited",
    "infer_object_at_place",
    "infer_container_moves_contents",
    "infer_object_in_container",
    "infer_unknown_location",
)

def infer_initial_location(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "initial_location" and first_location(structure, query.target) is not None:
        return "initial_location_found"
    return None
def infer_location_before_actor_action(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location_before_actor_action":
        return None
    actor = query_qualifier(query, "actor")
    if location_before_actor_action(structure, query.target, actor) is not None:
        return "location_before_actor_action_found"
    return None
def infer_location_before_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location_before_event":
        return None
    if temporal_event_location(structure, query, include_anchor=False) is not None:
        return "location_before_event_found"
    return "location_before_event_unknown"
def infer_location_after_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location_after_event":
        return None
    if temporal_event_location(structure, query, include_anchor=True) is not None:
        return "location_after_event_found"
    return "location_after_event_unknown"
def infer_counterfactual_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "counterfactual_location":
        return None
    if counterfactual_location(structure, query) is not None:
        return "counterfactual_location_found"
    return "counterfactual_location_unknown"
def infer_polar_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "polar_location":
        return None
    status = polar_location_status(structure, query)
    if status is None:
        return "polar_location_unknown"
    return "polar_location_true" if status else "polar_location_false"
def infer_same_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "same_location":
        return None
    status = same_location_status(structure, query)
    if status is None:
        return "same_location_unknown"
    return "same_location_true" if status else "same_location_false"
def infer_places_visited(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "places_visited" and places_visited(structure, query.target):
        return "places_visited"
    return None
def infer_object_at_place(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "location" and has_state_left(structure, "at", query.target):
        return "object_at_place"
    return None
def infer_container_moves_contents(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location":
        return None
    place, containers = location_path(structure, query.target)
    if place is not None and containers:
        return "container_moves_contents"
    return None
def infer_object_in_container(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location":
        return None
    place, containers = location_path(structure, query.target)
    if place is None and containers:
        return "object_in_container"
    return None
def infer_unknown_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location":
        return None
    if object_exists_value(structure, query.target) == "不存在":
        return None
    if has_state_left(structure, "at", query.target):
        return None
    place, containers = location_path(structure, query.target)
    if place is not None or containers:
        return None
    return "location_unknown"
