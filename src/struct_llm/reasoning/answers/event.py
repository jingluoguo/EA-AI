from __future__ import annotations

from ...errors import ParseError
from ...structure import Structure
from ..selectors import *

__all__ = (
    "answer_event_actor",
    "answer_latest_event_actor",
    "answer_earliest_event_actor",
    "answer_actor_handles_item",
    "answer_latest_actor_for_item",
    "answer_events_after_event",
    "answer_actions_by_actors",
)

def answer_event_actor(structure: Structure) -> str | None:
    if "event_actor_matches" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = actor_for_event_query(structure, query)
    item = query_qualifier(query, "item")
    if query.target == "put_in":
        holder = query_qualifier(query, "holder")
        return f"{actor}把{item}放进{holder}。"
    if query.target == "take_out":
        source = query_qualifier(query, "source")
        return f"{actor}把{item}从{source}取出。"
    return f"{actor}执行了{query.target}。"
def answer_latest_event_actor(structure: Structure) -> str | None:
    if "latest_event_actor_matches" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = latest_actor_for_event_query(structure, query)
    item = query_qualifier(query, "item")
    if query.target == "put_in":
        holder = query_qualifier(query, "holder")
        return f"最后是{actor}把{item}放进{holder}。"
    if query.target == "take_out":
        source = query_qualifier(query, "source")
        return f"最后是{actor}把{item}从{source}取出。"
    return f"最后是{actor}执行了{query.target}。"
def answer_earliest_event_actor(structure: Structure) -> str | None:
    if "earliest_event_actor_matches" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = earliest_actor_for_event_query(structure, query)
    item = query_qualifier(query, "item")
    if query.target == "put_in":
        holder = query_qualifier(query, "holder")
        return f"最先是{actor}把{item}放进{holder}。"
    if query.target == "take_out":
        source = query_qualifier(query, "source")
        return f"最先是{actor}把{item}从{source}取出。"
    return f"最先是{actor}执行了{query.target}。"
def answer_actor_handles_item(structure: Structure) -> str | None:
    if "actor_handles_item" not in set(structure.rules):
        return None
    query = require_query(structure)
    frame = latest_frame_with_role(structure, "handle", "theme", query.target)
    actor = required_frame_role(frame, "actor")
    return f"{actor}拿的{query.target}。"
def answer_latest_actor_for_item(structure: Structure) -> str | None:
    if "latest_actor_handles_item" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = latest_actor_for_item(structure, query.target)
    if actor is None:
        raise ParseError(f"Expected latest actor for {query.target}.")
    return f"最后是{actor}处理过{query.target}。"
def answer_events_after_event(structure: Structure) -> str | None:
    if "events_after_event" not in set(structure.rules):
        return None
    query = require_query(structure)
    descriptions = events_after_query(structure, query)
    return f"之后发生了：{'；'.join(descriptions)}。"
def answer_actions_by_actors(structure: Structure) -> str | None:
    if "actor_actions" not in set(structure.rules):
        return None
    query = require_query(structure)
    actors = query_qualifier(query, "actors").split("|")
    parts = []
    for actor in actors:
        descriptions = action_descriptions_for_actor(structure, actor)
        if descriptions:
            parts.append(f"{actor}{'，'.join(descriptions)}")
        else:
            parts.append(f"{actor}没有已知动作")
    return "；".join(parts) + "。"
