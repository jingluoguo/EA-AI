from __future__ import annotations

from ...errors import ParseError
from ...structure import Frame, Query, Structure
from ...world.event_schema import frame_matches_qualifiers
from .common import query_qualifier, required_frame_role

__all__ = (
    "matches_clause_target",
    "frame_matches_query",
    "describe_frame_action",
    "describe_historical_frame",
    "summary_descriptions",
    "action_descriptions_for_actor",
    "latest_actor_for_item",
    "earliest_actor_for_item",
    "first_matching_event_frame",
    "has_actor_for_event_query",
    "actor_for_event_query",
    "has_latest_actor_for_event_query",
    "has_earliest_actor_for_event_query",
    "latest_actor_for_event_query",
    "earliest_actor_for_event_query",
    "events_after_query",
    "first_actor_action_frame",
    "temporal_anchor_frame",
    "frame_matches_counterfactual_exclusion",
)

def matches_clause_target(clause: str, target: str) -> bool:
    normalized_clause = clause.strip().rstrip("。！？!?")
    normalized_target = target.strip().rstrip("。！？!?")
    return normalized_clause == normalized_target or normalized_target in normalized_clause or normalized_clause in normalized_target
def frame_matches_query(frame: Frame, query: Query) -> bool:
    return frame_matches_qualifiers(frame, query.qualifiers)
def describe_frame_action(frame: Frame) -> str:
    if frame.frame_type == "put_in":
        return f"把{required_frame_role(frame, 'theme')}放进{required_frame_role(frame, 'goal')}"
    if frame.frame_type == "take_out":
        return f"把{required_frame_role(frame, 'theme')}从{required_frame_role(frame, 'source')}取出"
    if frame.frame_type == "move":
        return f"把{required_frame_role(frame, 'theme')}带到{required_frame_role(frame, 'goal')}"
    if frame.frame_type == "give":
        return f"把{required_frame_role(frame, 'theme')}交给{required_frame_role(frame, 'recipient')}"
    if frame.frame_type == "paint":
        return f"把{required_frame_role(frame, 'theme')}涂成{required_frame_role(frame, 'result')}"
    if frame.frame_type == "open":
        return f"打开{required_frame_role(frame, 'theme')}"
    if frame.frame_type == "close":
        return f"关闭{required_frame_role(frame, 'theme')}"
    return f"执行{frame.frame_type}"
def describe_historical_frame(frame: Frame) -> str:
    actor = frame.role("actor")
    if actor:
        return f"{actor}{describe_frame_action(frame)}"
    if frame.frame_type == "profile_name":
        return f"{required_frame_role(frame, 'subject')}叫{required_frame_role(frame, 'value')}"
    if frame.frame_type == "profile_like":
        return f"{required_frame_role(frame, 'subject')}喜欢{required_frame_role(frame, 'value')}"
    if frame.frame_type == "profile_dislike":
        return f"{required_frame_role(frame, 'subject')}不喜欢{required_frame_role(frame, 'value')}"
    if frame.frame_type == "say":
        return f"{required_frame_role(frame, 'speaker')}说{required_frame_role(frame, 'proposition')}"
    if frame.frame_type == "believe":
        return f"{required_frame_role(frame, 'person')}认为{required_frame_role(frame, 'proposition')}"
    if frame.frame_type == "because":
        return f"因为{required_frame_role(frame, 'cause')}，所以{required_frame_role(frame, 'effect')}"
    if frame.frame_type == "if_then":
        return f"如果{required_frame_role(frame, 'antecedent')}，就{required_frame_role(frame, 'consequent')}"
    if frame.frame_type == "move":
        return f"{required_frame_role(frame, 'theme')}被带到{required_frame_role(frame, 'goal')}"
    if frame.frame_type == "take_out":
        return f"{required_frame_role(frame, 'theme')}从{required_frame_role(frame, 'source')}被取出"
    if frame.frame_type == "be_in":
        return f"{required_frame_role(frame, 'theme')}在{required_frame_role(frame, 'goal')}里"
    if frame.frame_type == "not_in":
        return f"{required_frame_role(frame, 'theme')}不在{required_frame_role(frame, 'source')}里"
    return describe_frame_action(frame)
def summary_descriptions(structure: Structure) -> tuple[str, ...]:
    descriptions = [
        describe_historical_frame(frame)
        for frame in structure.frames
        if frame.frame_type != "handle"
    ]
    return tuple(dict.fromkeys(descriptions))
def action_descriptions_for_actor(structure: Structure, actor: str) -> tuple[str, ...]:
    descriptions: list[str] = []
    for frame in structure.frames:
        if frame.frame_type == "handle" or frame.role("actor") != actor:
            continue
        descriptions.append(describe_frame_action(frame))
    return tuple(descriptions)
def latest_actor_for_item(structure: Structure, target: str) -> str | None:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.role("theme") == target and frame.role("actor") is not None
    ]
    return matches[-1] if matches else None
def earliest_actor_for_item(structure: Structure, target: str) -> str | None:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.role("theme") == target and frame.role("actor") is not None
    ]
    return matches[0] if matches else None
def first_matching_event_frame(structure: Structure, query: Query) -> Frame | None:
    matches = [
        frame
        for frame in structure.frames
        if frame.frame_type == query.target and frame_matches_query(frame, query)
    ]
    return matches[0] if matches else None
def has_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True
def actor_for_event_query(structure: Structure, query: Query) -> str:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.frame_type == query.target
        and frame_matches_query(frame, query)
    ]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one actor for event query, got {len(matches)}.")
    return matches[0]
def has_latest_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        latest_actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True
def has_earliest_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        earliest_actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True
def latest_actor_for_event_query(structure: Structure, query: Query) -> str:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.frame_type == query.target
        and frame_matches_query(frame, query)
    ]
    if not matches:
        raise ParseError(f"Expected at least one {query.target} frame for latest actor query.")
    return matches[-1]
def earliest_actor_for_event_query(structure: Structure, query: Query) -> str:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.frame_type == query.target
        and frame_matches_query(frame, query)
    ]
    if not matches:
        raise ParseError(f"Expected at least one {query.target} frame for earliest actor query.")
    return matches[0]
def events_after_query(structure: Structure, query: Query) -> tuple[str, ...]:
    anchor = first_matching_event_frame(structure, query)
    if anchor is None:
        return ()
    frames = [
        frame
        for frame in structure.frames
        if frame.time > anchor.time and frame.frame_type != "handle"
    ]
    return tuple(describe_historical_frame(frame) for frame in frames)
def first_actor_action_frame(structure: Structure, target: str, actor: str) -> Frame | None:
    frames = [
        frame
        for frame in structure.frames
        if frame.frame_type != "handle" and frame.role("actor") == actor and frame.role("theme") == target
    ]
    if not frames:
        frames = [
            frame
            for frame in structure.frames
            if frame.role("actor") == actor and frame.role("theme") == target
        ]
    return frames[0] if frames else None
def temporal_anchor_frame(structure: Structure, query: Query) -> Frame | None:
    event_type = query_qualifier(query, "event")
    matches = [
        frame
        for frame in structure.frames
        if frame.frame_type == event_type
        and frame_matches_qualifiers(frame, query.qualifiers, ignored_keys=("anchor", "event"))
    ]
    return matches[0] if matches else None
def frame_matches_counterfactual_exclusion(frame: Frame, query: Query) -> bool:
    event = query_qualifier(query, "without_event")
    if frame.frame_type != event:
        return False
    return frame_matches_qualifiers(frame, query.qualifiers, ignored_keys=("without_event",))
