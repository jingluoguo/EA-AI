from __future__ import annotations

from dataclasses import dataclass

from ..capabilities import CognitiveCapabilities
from ..structure import Entity, Frame, Query, State

LAST_USER_UTTERANCE_STATE = "last_user_utterance"


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
    if query.intent in {"dialog_act", "compound"}:
        return ()
    target = query.target.strip()
    if not target or "$" in target or target == "multi":
        return ()
    return (
        State("focus_topic", "user", target, "working_memory"),
        State("focus_query_intent", "user", query.intent, "working_memory"),
    )


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
    return capabilities.evolve(memory_states=tuple(preserved))


def last_user_utterance(states: tuple[State, ...]) -> str:
    matches = [
        state.right
        for state in states
        if state.name == LAST_USER_UTTERANCE_STATE and state.left == "user" and state.right
    ]
    return matches[-1] if matches else ""
