from __future__ import annotations

from dataclasses import dataclass

from ..capabilities import CognitiveCapabilities
from ..structure import Entity, Frame, Query, State

LAST_USER_UTTERANCE_STATE = "last_user_utterance"
ONGOING_DIALOG_ACT_TARGETS = frozenset({"meal_suggestion"})


@dataclass(frozen=True)
class WorkingMemory:
    focus_entities: tuple[Entity, ...] = ()
    recent_frames: tuple[Frame, ...] = ()
    current_states: tuple[State, ...] = ()

    def with_focus(self, *entities: Entity) -> "WorkingMemory":
        return WorkingMemory(
            focus_entities=(*self.focus_entities, *entities),
            recent_frames=self.recent_frames,
            current_states=self.current_states,
        )

    def with_frame(self, frame: Frame) -> "WorkingMemory":
        return WorkingMemory(
            focus_entities=self.focus_entities,
            recent_frames=(*self.recent_frames, frame),
            current_states=self.current_states,
        )

    def with_states(self, *states: State) -> "WorkingMemory":
        return WorkingMemory(
            focus_entities=self.focus_entities,
            recent_frames=self.recent_frames,
            current_states=states,
        )


def focus_states_for_query(query: Query | None) -> tuple[State, ...]:
    if query is None:
        return ()
    if query.intent == "dialog_act":
        if query.target not in ONGOING_DIALOG_ACT_TARGETS:
            return ()
        states = [State("focus_dialog_act", "user", query.target, "working_memory")]
        preference = query_qualifier(query, "preference")
        if preference:
            states.append(State("focus_dialog_preference", query.target, preference, "working_memory"))
        return tuple(states)
    if query.intent in {"compound", "profile"}:
        return ()
    target = query.target.strip()
    if not target or "$" in target or target == "multi":
        return ()
    return (
        State("focus_topic", "user", target, "working_memory"),
        State("focus_query_intent", "user", query.intent, "working_memory"),
        *focus_condition_states_for_query(query),
    )


def focus_states_for_frames(frames: tuple[Frame, ...]) -> tuple[State, ...]:
    latest_topic = ""
    latest_condition_topic = ""
    for frame in frames:
        if frame.frame_type == "condition":
            condition_topic = condition_frame_focus_topic(frame)
            if condition_topic:
                latest_condition_topic = condition_topic
            continue
        if frame.frame_type == "material":
            topic = frame_focus_topic(frame)
            if topic:
                latest_topic = topic
    states: list[State] = []
    if latest_topic:
        states.append(State("focus_topic", "user", latest_topic, "working_memory"))
    if latest_condition_topic:
        states.append(State("focus_condition_topic", "user", latest_condition_topic, "working_memory"))
    return tuple(states)


def focus_condition_states_for_query(query: Query | None) -> tuple[State, ...]:
    if query is None or query.intent != "why":
        return ()
    target = query.target.strip()
    if not target or "$" in target or target == "multi":
        return ()
    return (State("focus_condition_topic", "user", target, "working_memory"),)


def condition_frame_focus_topic(frame: Frame) -> str:
    theme = frame.role("theme")
    result = frame.role("result")
    if not theme or not result:
        return ""
    return f"{theme}{result}"


def frame_focus_topic(frame: Frame) -> str:
    for role_name in ("theme", "item", "object", "subject"):
        value = frame.role(role_name)
        if value:
            return value
    return ""


def capabilities_with_last_user_utterance(
    capabilities: CognitiveCapabilities,
    text: str,
) -> CognitiveCapabilities:
    cleaned = text.strip()
    if not cleaned:
        return capabilities
    preserved = tuple(
        state
        for state in capabilities.memory_states
        if state.name != LAST_USER_UTTERANCE_STATE
    )
    return capabilities.evolve(
        memory_states=(
            *preserved,
            State(LAST_USER_UTTERANCE_STATE, "user", cleaned, "working_memory"),
        )
    )


def capabilities_with_working_turn(
    capabilities: CognitiveCapabilities,
    text: str,
    states: tuple[State, ...],
    query: Query | None = None,
    frames: tuple[Frame, ...] = (),
) -> CognitiveCapabilities:
    preserved = [
        state
        for state in capabilities.memory_states
        if state.name != LAST_USER_UTTERANCE_STATE
    ]
    for state in states:
        if state.name == LAST_USER_UTTERANCE_STATE:
            continue
        capabilities.apply_state(preserved, state)
    for state in focus_states_for_query(query):
        capabilities.apply_state(preserved, state)
    cleaned = text.strip()
    if cleaned:
        preserved.append(State(LAST_USER_UTTERANCE_STATE, "user", cleaned, "working_memory"))
    existing_frames = tuple(capabilities.memory_frames)
    if frames and frames[: len(existing_frames)] == existing_frames:
        new_frames = frames[len(existing_frames) :]
    else:
        new_frames = frames
    for frame in new_frames:
        for state in capabilities.states_from_frame(frame):
            capabilities.apply_state(preserved, state)
    for state in focus_states_for_frames(new_frames):
        capabilities.apply_state(preserved, state)
    preserved_frames = (*existing_frames, *new_frames) if new_frames else existing_frames
    return capabilities.evolve(memory_states=tuple(preserved), memory_frames=preserved_frames)


def last_user_utterance(states: tuple[State, ...]) -> str:
    matches = [
        state.right
        for state in states
        if state.name == LAST_USER_UTTERANCE_STATE and state.left == "user" and state.right
    ]
    return matches[-1] if matches else ""


def query_qualifier(query: Query, key: str) -> str:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in query.qualifiers if qualifier.startswith(prefix)]
    return matches[-1] if matches else ""
