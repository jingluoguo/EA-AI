from __future__ import annotations

from ...structure import Frame, State, Structure
from ...world.state import apply_state, states_from_frame
from .common import object_exists_value, object_is_known, required_frame_role, state_for_left, state_for_left_or_none, structure_with_states
from .event import matches_clause_target
from .location import container_chain_text, location_path, location_phrase

__all__ = (
    "belief_location",
    "belief_states",
    "claim_speakers",
    "belief_sources",
    "states_from_proposition",
    "explanation_for_target",
    "explain_location_target",
    "explain_owner_target",
    "explain_color_target",
    "contradictions",
    "source_for_view_frame",
    "conflicting_fact_phrase",
    "fact_phrase",
)

def belief_location(structure: Structure, person: str, target: str) -> tuple[str | None, tuple[str, ...]] | None:
    states = belief_states(structure, person)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, target)
    if place is None and not containers:
        return None
    return place, containers
def belief_states(structure: Structure, person: str) -> list[State]:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        if frame.frame_type != "believe" or frame.role("person") != person:
            continue
        frame_states = scoped_states_for_frame(structure, frame, "belief")
        if not frame_states:
            frame_states = states_from_proposition(structure, required_frame_role(frame, "proposition"))
        for state in frame_states:
            apply_state(states, state)
    return states
def claim_speakers(structure: Structure, proposition: str) -> tuple[str, ...]:
    normalized = proposition.strip().rstrip("。！？!?")
    speakers = [
        required_frame_role(frame, "speaker")
        for frame in structure.frames
        if frame.frame_type == "say" and matches_clause_target(required_frame_role(frame, "proposition"), normalized)
    ]
    return tuple(dict.fromkeys(speakers))
def belief_sources(structure: Structure, proposition: str) -> tuple[str, ...]:
    normalized = proposition.strip().rstrip("。！？!?")
    believers = [
        required_frame_role(frame, "person")
        for frame in structure.frames
        if frame.frame_type == "believe" and matches_clause_target(required_frame_role(frame, "proposition"), normalized)
    ]
    return tuple(dict.fromkeys(believers))
def states_from_proposition(structure: Structure, proposition: str) -> tuple[State, ...]:
    normalized = proposition.strip().rstrip("。！？!?")
    scoped = scoped_states_for_proposition(structure, normalized)
    if scoped:
        return scoped

    if normalized.endswith("不存在"):
        return (State("exists", normalized.removesuffix("不存在"), "不存在"),)
    if normalized.endswith("存在"):
        return (State("exists", normalized.removesuffix("存在"), "存在"),)

    if "在" not in proposition:
        return ()
    left, right = proposition.split("在", 1)
    left = left.strip()
    right = right.strip().rstrip("。！？!?")
    if not left or not right:
        return ()

    if "的" in right:
        place, container = right.split("的", 1)
        container = container.removesuffix("里面").removesuffix("里边").removesuffix("里头")
        container = container.removesuffix("内部").removesuffix("里").removesuffix("内").removesuffix("中")
        if place and container:
            return (
                State("at", container, place),
                State("in", left, container),
            )

    place_names = {entity.name for entity in structure.entities if entity.role == "place"}
    for suffix in ("里面", "里边", "里头", "内部", "里", "内", "中"):
        if right.endswith(suffix) and len(right) > len(suffix):
            right = right[: -len(suffix)]
            return (State("in", left, right),)
    state_name = "at" if right in place_names else "in"
    return (State(state_name, left, right),)
def scoped_states_for_frame(structure: Structure, frame: Frame, kind: str | None = None) -> tuple[State, ...]:
    return tuple(
        scoped.state
        for scoped in structure.scoped_states
        if scoped.scope == frame.frame_id and (kind is None or scoped.kind == kind)
    )
def scoped_states_for_proposition(structure: Structure, proposition: str) -> tuple[State, ...]:
    normalized = proposition.strip().rstrip("。！？!?")
    states: list[State] = []
    seen: set[tuple[str, str, str]] = set()
    for scoped in structure.scoped_states:
        if not matches_clause_target(scoped.proposition, normalized):
            continue
        signature = (scoped.state.name, scoped.state.left, scoped.state.right)
        if signature in seen:
            continue
        seen.add(signature)
        states.append(scoped.state)
    return tuple(states)
def explanation_for_target(structure: Structure, target: str) -> str | None:
    target = target.strip().rstrip("。！？!?")

    target_states = states_from_proposition(structure, target)
    because_matches = [
        required_frame_role(frame, "cause")
        for frame in structure.frames
        if frame.frame_type == "because" and (
            matches_clause_target(required_frame_role(frame, "effect"), target)
            or states_cover_target(scoped_states_for_frame(structure, frame, "effect"), target_states)
        )
    ]
    if because_matches:
        return f"因为{because_matches[-1]}。"

    target_state_names = {state.name for state in target_states}
    if target_state_names & {"in", "at"}:
        location = explain_location_target(structure, target, target_states)
        if location is not None:
            return location

    if "owner" in target_state_names or "拥有" in target:
        owner = explain_owner_target(structure, target, target_states)
        if owner is not None:
            return owner

    if "color" in target_state_names or "颜色" in target or "是" in target:
        color = explain_color_target(structure, target, target_states)
        if color is not None:
            return color

    return None
def states_cover_target(candidates: tuple[State, ...], target_states: tuple[State, ...]) -> bool:
    if not candidates or not target_states:
        return False
    candidate_signatures = {(state.name, state.left, state.right) for state in candidates}
    return all((state.name, state.left, state.right) in candidate_signatures for state in target_states)
def explain_location_target(structure: Structure, target: str, target_states: tuple[State, ...] = ()) -> str | None:
    object_name = location_object_from_states(target_states) or target.split("在", 1)[0]
    if not object_name:
        return None
    place, containers = location_path(structure, object_name)
    if place is None and not containers:
        return None
    if containers:
        if place is not None:
            return f"因为{object_name}在{container_chain_text(containers)}，而且{containers[-1]}在{place}。"
        return f"因为{object_name}在{container_chain_text(containers)}。"
    if place is not None:
        at_frames = [frame for frame in structure.frames if frame.frame_type == "move" and frame.role("theme") == object_name]
        if at_frames:
            return f"因为{object_name}被带到{place}。"
        return f"因为{object_name}在{place}。"
    return None
def location_object_from_states(states: tuple[State, ...]) -> str | None:
    location_states = [state for state in states if state.name in {"in", "at"}]
    if not location_states:
        return None
    right_values = {state.right for state in location_states}
    for state in location_states:
        if state.left not in right_values:
            return state.left
    return location_states[-1].left
def explain_owner_target(structure: Structure, target: str, target_states: tuple[State, ...] = ()) -> str | None:
    object_name = next((state.left for state in target_states if state.name == "owner"), "")
    if not object_name:
        object_name = target.split("拥有", 1)[0]
    owner = state_for_left(structure, "owner", object_name) if any(
        state.name == "owner" and state.left == object_name for state in structure.states
    ) else None
    if owner is None:
        return None
    return f"因为{owner.right}拥有{owner.left}。"
def explain_color_target(structure: Structure, target: str, target_states: tuple[State, ...] = ()) -> str | None:
    object_name = next((state.left for state in target_states if state.name == "color"), "")
    if not object_name:
        object_name = target.split("是", 1)[0].replace("颜色", "")
    color = state_for_left(structure, "color", object_name) if any(
        state.name == "color" and state.left == object_name for state in structure.states
    ) else None
    if color is None:
        return None
    return f"因为{color.left}是{color.right}。"
def contradictions(structure: Structure) -> tuple[str, ...]:
    found: list[str] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        source = source_for_view_frame(frame)
        if source is None:
            continue
        actor, verb = source
        proposition = required_frame_role(frame, "proposition")
        frame_states = scoped_states_for_frame(structure, frame)
        if not frame_states:
            frame_states = states_from_proposition(structure, proposition)
        for state in frame_states:
            fact = conflicting_fact_phrase(structure, state)
            if fact is not None:
                found.append(f"{actor}{verb}{proposition}，但事实是{fact}")
                break
    return tuple(dict.fromkeys(found))
def source_for_view_frame(frame: Frame) -> tuple[str, str] | None:
    if frame.frame_type == "say":
        return required_frame_role(frame, "speaker"), "说"
    if frame.frame_type == "believe":
        return required_frame_role(frame, "person"), "认为"
    return None
def conflicting_fact_phrase(structure: Structure, claimed: State) -> str | None:
    if claimed.name == "exists":
        actual_value = object_exists_value(structure, claimed.left)
        if actual_value is not None:
            return None if actual_value == claimed.right else fact_phrase(structure, State("exists", claimed.left, actual_value))
        if claimed.right == "不存在" and object_is_known(structure, claimed.left):
            return fact_phrase(structure, State("exists", claimed.left, "存在"))
        return None

    if claimed.name == "not_in":
        actual = state_for_left_or_none(structure, "in", claimed.left)
        if actual is not None and (not claimed.right or actual.right == claimed.right):
            return fact_phrase(structure, State("in", claimed.left, actual.right))
        return None

    actual = state_for_left_or_none(structure, claimed.name, claimed.left)
    if actual is not None:
        return None if actual.right == claimed.right else fact_phrase(structure, actual)

    if claimed.name in {"in", "at"}:
        place, containers = location_path(structure, claimed.left)
        if claimed.name == "in" and claimed.right in containers:
            return None
        if claimed.name == "at" and place == claimed.right:
            return None
        if place is not None or containers:
            return f"{claimed.left}{location_phrase(place, containers)}"
    return None
def fact_phrase(structure: Structure, state: State) -> str:
    if state.name in {"in", "at"}:
        place, containers = location_path(structure, state.left)
        if place is not None or containers:
            return f"{state.left}{location_phrase(place, containers)}"
    if state.name == "owner":
        return f"{state.right}拥有{state.left}"
    if state.name == "color":
        return f"{state.left}是{state.right}"
    if state.name == "exists":
        return f"{state.left}{state.right}"
    return f"{state.name}({state.left},{state.right})"
