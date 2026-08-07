from __future__ import annotations

from ...structure import Structure
from ..selectors import *

__all__ = (
    "infer_event_actor_matches",
    "infer_latest_event_actor_matches",
    "infer_earliest_event_actor_matches",
    "infer_actor_handles_item",
    "infer_latest_actor_for_item",
    "infer_events_after_event",
    "infer_actions_by_actors",
)

def infer_event_actor_matches(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "actor_for_event" and has_actor_for_event_query(structure, query):
        return "event_actor_matches"
    return None
def infer_latest_event_actor_matches(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "latest_actor_for_event" and has_latest_actor_for_event_query(structure, query):
        return "latest_event_actor_matches"
    return None
def infer_earliest_event_actor_matches(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "earliest_actor_for_event" and has_earliest_actor_for_event_query(structure, query):
        return "earliest_event_actor_matches"
    return None
def infer_actor_handles_item(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "actor_for_item" and has_frame_with_role(
        structure, "handle", "theme", query.target
    ):
        return "actor_handles_item"
    return None
def infer_latest_actor_for_item(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "latest_actor_for_item" and latest_actor_for_item(structure, query.target):
        return "latest_actor_handles_item"
    return None
def infer_events_after_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "events_after_event":
        return None
    if events_after_query(structure, query):
        return "events_after_event"
    return None
def infer_actions_by_actors(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "actions_by_actors":
        return None
    actors = query_qualifier(query, "actors").split("|")
    if any(action_descriptions_for_actor(structure, actor) for actor in actors):
        return "actor_actions"
    return None
