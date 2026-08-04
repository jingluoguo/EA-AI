from __future__ import annotations

from .capabilities import Answerer, RuleInferer
from .errors import ParseError
from .structure import Frame, Query, State, Structure


def infer_rules(
    structure: Structure,
    inferers: tuple[RuleInferer, ...] | None = None,
) -> tuple[str, ...]:
    if structure.query is None:
        return ()

    rules: list[str] = []
    for inferer in inferers or DEFAULT_RULE_INFERERS:
        rule = inferer(structure)
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def infer_event_actor_matches(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "actor_for_event" and has_actor_for_event_query(structure, query):
        return "event_actor_matches"
    return None


def infer_actor_handles_item(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "actor_for_item" and has_frame_with_role(
        structure, "handle", "theme", query.target
    ):
        return "actor_handles_item"
    return None


def infer_holder_contains_things(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contents" and contents_in_holder(structure, query.target):
        return "holder_contains_things"
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
    in_states = [state for state in structure.states if state.name == "in" and state.left == query.target]
    if any(has_state_left(structure, "at", state.right) for state in in_states):
        return "container_moves_contents"
    return None


def infer_transfer_changes_owner(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "owner" and has_state_left(structure, "owner", query.target):
        return "transfer_changes_owner"
    return None


def infer_paint_changes_color(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "color" and has_state_left(structure, "color", query.target):
        return "paint_changes_color"
    return None


def answer_from_structure(
    structure: Structure,
    answerers: tuple[Answerer, ...] | None = None,
) -> str:
    for answerer in answerers or DEFAULT_ANSWERERS:
        answer = answerer(structure)
        if answer is not None:
            return answer

    raise ParseError(f"No rule can answer structure: {structure.linearize()}")


def answer_event_actor(structure: Structure) -> str | None:
    if "event_actor_matches" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = actor_for_event_query(structure, query)
    item = query_qualifier(query, "item")
    holder = query_qualifier(query, "holder")
    if query.target == "put_in":
        return f"{actor}把{item}放进{holder}。"
    return f"{actor}执行了{query.target}。"


def answer_actor_handles_item(structure: Structure) -> str | None:
    if "actor_handles_item" not in set(structure.rules):
        return None
    query = require_query(structure)
    frame = latest_frame_with_role(structure, "handle", "theme", query.target)
    actor = required_frame_role(frame, "actor")
    return f"{actor}拿的{query.target}。"


def answer_holder_contains_things(structure: Structure) -> str | None:
    if "holder_contains_things" not in set(structure.rules):
        return None
    query = require_query(structure)
    contents = contents_in_holder(structure, query.target)
    return f"{query.target}里至少有{join_names(contents)}。"


def answer_object_at_place(structure: Structure) -> str | None:
    if "object_at_place" not in set(structure.rules):
        return None
    query = require_query(structure)
    state = state_for_left(structure, "at", query.target)
    return f"{query.target}在{state.right}。"


def answer_container_moves_contents(structure: Structure) -> str | None:
    if "container_moves_contents" not in set(structure.rules):
        return None
    query = structure.query
    in_state = state_for_left(structure, "in", query.target) if query else only_state(structure, "in")
    at_state = state_for_left(structure, "at", in_state.right)
    item = in_state.left
    container = in_state.right
    place = at_state.right
    return f"{item}在{place}的{container}里。"


def answer_transfer_changes_owner(structure: Structure) -> str | None:
    if "transfer_changes_owner" not in set(structure.rules):
        return None
    query = structure.query
    owner = state_for_left(structure, "owner", query.target) if query else only_state(structure, "owner")
    item = owner.left
    receiver = owner.right
    return f"{receiver}拥有{item}。"


def answer_paint_changes_color(structure: Structure) -> str | None:
    if "paint_changes_color" not in set(structure.rules):
        return None
    query = structure.query
    color_relation = state_for_left(structure, "color", query.target) if query else only_state(structure, "color")
    item = color_relation.left
    color = color_relation.right
    return f"{item}是{color}。"


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
    return contents


def has_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True


def actor_for_event_query(structure: Structure, query: Query) -> str:
    item = query_qualifier(query, "item")
    holder = query_qualifier(query, "holder")
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.frame_type == query.target
        and frame.role("theme") == item
        and frame.role("goal") == holder
    ]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one actor for event query, got {len(matches)}.")
    return matches[0]


def query_qualifier(query: Query, key: str) -> str:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in query.qualifiers if qualifier.startswith(prefix)]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one query qualifier {key}, got {len(matches)}.")
    return matches[0]


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


def has_state_left(structure: Structure, name: str, left: str) -> bool:
    return any(state.name == name and state.left == left for state in structure.states)


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


def require_query(structure: Structure) -> Query:
    if structure.query is None:
        raise ParseError("Expected query in structure.")
    return structure.query


def join_names(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return "和".join(names)


DEFAULT_RULE_INFERERS: tuple[RuleInferer, ...] = (
    infer_event_actor_matches,
    infer_actor_handles_item,
    infer_holder_contains_things,
    infer_object_at_place,
    infer_container_moves_contents,
    infer_transfer_changes_owner,
    infer_paint_changes_color,
)
DEFAULT_ANSWERERS: tuple[Answerer, ...] = (
    answer_event_actor,
    answer_actor_handles_item,
    answer_holder_contains_things,
    answer_object_at_place,
    answer_container_moves_contents,
    answer_transfer_changes_owner,
    answer_paint_changes_color,
)
