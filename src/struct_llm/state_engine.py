from __future__ import annotations

from .capabilities import StateProjector, StateReducer
from .structure import Frame, State


def states_from_frame(
    frame: Frame,
    projectors: tuple[StateProjector, ...] | None = None,
) -> tuple[State, ...]:
    states: list[State] = []
    for projector in projectors or DEFAULT_STATE_PROJECTORS:
        states.extend(projector(frame))
    return tuple(states)


def put_in_state(frame: Frame) -> tuple[State, ...]:
    if frame.frame_type == "put_in":
        theme = frame.role("theme") or ""
        goal = frame.role("goal") or ""
        return (State("in", theme, goal, frame.frame_id),)
    return ()


def move_state(frame: Frame) -> tuple[State, ...]:
    if frame.frame_type == "move":
        theme = frame.role("theme") or ""
        goal = frame.role("goal") or ""
        return (State("at", theme, goal, frame.frame_id),)
    return ()


def give_state(frame: Frame) -> tuple[State, ...]:
    if frame.frame_type == "give":
        theme = frame.role("theme") or ""
        recipient = frame.role("recipient") or ""
        return (State("owner", theme, recipient, frame.frame_id),)
    return ()


def paint_state(frame: Frame) -> tuple[State, ...]:
    if frame.frame_type == "paint":
        theme = frame.role("theme") or ""
        result = frame.role("result") or ""
        return (State("color", theme, result, frame.frame_id),)
    return ()


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
    if state.name in {"in", "at", "owner", "color"}:
        states[:] = [
            existing
            for existing in states
            if not (existing.name == state.name and existing.left == state.left)
        ]
        states.append(state)
        return True
    return False


def materialize_relations(states: list[State]):
    return tuple(state.to_relation() for state in states)


def materialize_events(frames: list[Frame]):
    return tuple(frame.to_event() for frame in frames)


DEFAULT_STATE_PROJECTORS: tuple[StateProjector, ...] = (
    put_in_state,
    move_state,
    give_state,
    paint_state,
)
DEFAULT_STATE_REDUCERS: tuple[StateReducer, ...] = (overwrite_current_state,)
