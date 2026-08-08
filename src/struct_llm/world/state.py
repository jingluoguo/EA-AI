from __future__ import annotations

from ..capabilities import StateProjector, StateReducer
from .event_schema import states_for_frame_schema
from ..structure import Frame, State


def states_from_frame(
    frame: Frame,
    projectors: tuple[StateProjector, ...] | None = None,
) -> tuple[State, ...]:
    states: list[State] = []
    for projector in projectors or DEFAULT_STATE_PROJECTORS:
        states.extend(projector(frame))
    return tuple(states)


def schema_state_projector(frame: Frame) -> tuple[State, ...]:
    return states_for_frame_schema(frame)


def apply_state(
    states: list[State],
    state: State,
    reducers: tuple[StateReducer, ...] | None = None,
) -> None:
    for reducer in reducers or DEFAULT_STATE_REDUCERS:
        if reducer(states, state):
            return
    states.append(state)


def overwrite_current_state(states: list[State], state: State) -> bool:
    if state.name == "exists" and state.right == "不存在":
        states[:] = [
            existing
            for existing in states
            if existing.left != state.left and existing.right != state.left
        ]
        states.append(state)
        return True

    if state.name == "exists":
        states[:] = [
            existing
            for existing in states
            if not (existing.name == "exists" and existing.left == state.left)
        ]
        states.append(state)
        return True

    if state.name == "not_in":
        states[:] = [
            existing
            for existing in states
            if not (
                existing.name == "in"
                and existing.left == state.left
                and (not state.right or existing.right == state.right)
            )
        ]
        return True

    if state.name in {"in", "at", "owner", "color", "access"}:
        states[:] = [
            existing
            for existing in states
            if not (
                (existing.name == state.name and existing.left == state.left)
                or (existing.name == "exists" and existing.left == state.left and existing.right == "不存在")
            )
        ]
        states.append(state)
        return True
    if state.name == "name":
        states[:] = [
            existing
            for existing in states
            if not (existing.name == state.name and existing.left == state.left)
        ]
        states.append(state)
        return True
    if state.name in {"focus_topic", "focus_query_intent", "focus_dialog_act", "focus_dialog_preference"}:
        states[:] = [
            existing
            for existing in states
            if not (existing.name == state.name and existing.left == state.left)
        ]
        states.append(state)
        return True
    if state.name in {"likes", "dislikes"}:
        opposite = "dislikes" if state.name == "likes" else "likes"
        states[:] = [
            existing
            for existing in states
            if not (
                existing.left == state.left
                and existing.right == state.right
                and existing.name in {state.name, opposite}
            )
        ]
        states.append(state)
        return True
    return False


def materialize_relations(states: list[State]):
    return tuple(state.to_relation() for state in states)


def materialize_events(frames: list[Frame]):
    return tuple(frame.to_event() for frame in frames)


DEFAULT_STATE_PROJECTORS: tuple[StateProjector, ...] = (
    schema_state_projector,
)
DEFAULT_STATE_REDUCERS: tuple[StateReducer, ...] = (overwrite_current_state,)
