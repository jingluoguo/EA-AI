from __future__ import annotations

from ...errors import ParseError
from ...structure import Structure
from ..selectors import *

__all__ = (
    "answer_initial_location",
    "answer_location_before_actor_action",
    "answer_location_before_event",
    "answer_location_before_event_unknown",
    "answer_location_after_event",
    "answer_location_after_event_unknown",
    "answer_counterfactual_location",
    "answer_counterfactual_location_unknown",
    "answer_polar_location_true",
    "answer_polar_location_false",
    "answer_polar_location_unknown",
    "answer_same_location_true",
    "answer_same_location_false",
    "answer_same_location_unknown",
    "answer_places_visited",
    "answer_object_at_place",
    "answer_container_moves_contents",
    "answer_object_in_container",
    "answer_unknown_location",
)

def answer_initial_location(structure: Structure) -> str | None:
    if "initial_location_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = first_location(structure, query.target)
    if location is None:
        raise ParseError(f"Expected initial location for {query.target}.")
    place, containers = location
    return f"{query.target}最开始{location_phrase(place, containers)}。"
def answer_location_before_actor_action(structure: Structure) -> str | None:
    if "location_before_actor_action_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = query_qualifier(query, "actor")
    location = location_before_actor_action(structure, query.target, actor)
    if location is None:
        raise ParseError(f"Expected location before {actor} action for {query.target}.")
    place, containers = location
    return f"{actor}操作之前，{query.target}{location_phrase(place, containers)}。"
def answer_location_before_event(structure: Structure) -> str | None:
    if "location_before_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = temporal_event_location(structure, query, include_anchor=False)
    if location is None:
        raise ParseError(f"Expected location before event for {query.target}.")
    place, containers = location
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之前，{query.target}{location_phrase(place, containers)}。"
def answer_location_before_event_unknown(structure: Structure) -> str | None:
    if "location_before_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之前在哪里。"
def answer_location_after_event(structure: Structure) -> str | None:
    if "location_after_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = temporal_event_location(structure, query, include_anchor=True)
    if location is None:
        raise ParseError(f"Expected location after event for {query.target}.")
    place, containers = location
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之后，{query.target}{location_phrase(place, containers)}。"
def answer_location_after_event_unknown(structure: Structure) -> str | None:
    if "location_after_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之后在哪里。"
def answer_counterfactual_location(structure: Structure) -> str | None:
    if "counterfactual_location_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = counterfactual_location(structure, query)
    if location is None:
        raise ParseError(f"Expected counterfactual location for {query.target}.")
    place, containers = location
    return f"如果没有这个事件，{query.target}会{location_phrase(place, containers)}。"
def answer_counterfactual_location_unknown(structure: Structure) -> str | None:
    if "counterfactual_location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"如果没有这个事件，不知道{query.target}会在哪里。"
def answer_polar_location_true(structure: Structure) -> str | None:
    if "polar_location_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    if kind == "at":
        return f"是，{query.target}在{expected}。"
    return f"是，{query.target}在{expected}里。"
def answer_polar_location_false(structure: Structure) -> str | None:
    if "polar_location_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    actual = location_path(structure, query.target)
    if kind == "at":
        place, containers = actual
        if place is None and not containers:
            return f"不是，不知道{query.target}在哪里。"
        return f"不是，{query.target}{location_phrase(place, containers)}。"
    place, containers = actual
    if place is None and not containers:
        return f"不是，不知道{query.target}在哪里。"
    return f"不是，{query.target}{location_phrase(place, containers)}。"
def answer_polar_location_unknown(structure: Structure) -> str | None:
    if "polar_location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    if kind == "at":
        return f"不知道{query.target}是不是在{expected}。"
    return f"不知道{query.target}是不是在{expected}里。"
def answer_same_location_true(structure: Structure) -> str | None:
    if "same_location_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    return f"是，{left}和{right}在同一个地方。"
def answer_same_location_false(structure: Structure) -> str | None:
    if "same_location_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    left_phrase = describe_object_location(structure, left)
    right_phrase = describe_object_location(structure, right)
    return f"不是，{left}{left_phrase}，{right}{right_phrase}。"
def answer_same_location_unknown(structure: Structure) -> str | None:
    if "same_location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    return f"不知道{left}和{right}是不是在同一个地方。"
def answer_places_visited(structure: Structure) -> str | None:
    if "places_visited" not in set(structure.rules):
        return None
    query = require_query(structure)
    places = places_visited(structure, query.target)
    return f"{query.target}经过了{join_names(places)}。"
def answer_object_at_place(structure: Structure) -> str | None:
    if "object_at_place" not in set(structure.rules):
        return None
    query = require_query(structure)
    state = state_for_left(structure, "at", query.target)
    return f"{query.target}在{state.right}。"
def answer_container_moves_contents(structure: Structure) -> str | None:
    if "container_moves_contents" not in set(structure.rules):
        return None
    query = require_query(structure)
    place, containers = location_path(structure, query.target)
    if place is None:
        raise ParseError(f"Expected place for container location of {query.target}.")
    return f"{query.target}在{place}的{container_chain_text(containers)}。"
def answer_object_in_container(structure: Structure) -> str | None:
    if "object_in_container" not in set(structure.rules):
        return None
    query = require_query(structure)
    _, containers = location_path(structure, query.target)
    return f"{query.target}在{container_chain_text(containers)}。"
def answer_unknown_location(structure: Structure) -> str | None:
    if "location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}在哪里。"
