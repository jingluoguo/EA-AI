from __future__ import annotations

from ..capabilities import CognitiveCapabilities
from ..errors import ParseError
from ..structure import Frame, Role, State, Structure


def expand_conditionals(structure: Structure, capabilities: CognitiveCapabilities) -> Structure:
    frames = list(structure.frames)
    states = list(structure.states)
    known_signatures = {frame_signature(frame) for frame in frames}
    next_time = max((frame.time for frame in frames), default=0) + 1

    for _ in range(20):
        changed = False
        for rule in [frame for frame in frames if frame.frame_type == "if_then"]:
            antecedent = required_frame_role(rule, "antecedent")
            consequent = required_frame_role(rule, "consequent")
            if not statement_satisfied(antecedent, frames, states, capabilities):
                continue
            parsed = capabilities.parse_statement(consequent)
            if parsed is None:
                continue
            _, consequence_frames = parsed
            for consequence in consequence_frames:
                signature = frame_signature(consequence)
                if signature in known_signatures:
                    continue
                timed = retime_frame(consequence, next_time)
                next_time += 1
                frames.append(timed)
                known_signatures.add(signature)
                for state in capabilities.states_from_frame(timed):
                    capabilities.apply_state(states, state)
                changed = True
        if not changed:
            break

    return structure_with_frames_states(structure, frames, states)


def statement_satisfied(
    sentence: str,
    frames: list[Frame],
    states: list[State],
    capabilities: CognitiveCapabilities,
) -> bool:
    parsed = capabilities.parse_statement(sentence)
    if parsed is None:
        return False
    _, expected_frames = parsed
    current_frame_signatures = {frame_signature(frame) for frame in frames}
    current_state_signatures = {state_signature(state) for state in states}
    for frame in expected_frames:
        projected_states = capabilities.states_from_frame(frame)
        if frame.frame_type in {"be_in", "not_in"} and projected_states:
            if all(state_signature(state) in current_state_signatures for state in projected_states):
                continue
        if frame_signature(frame) not in current_frame_signatures:
            return False
    return True


def frame_signature(frame: Frame) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        frame.frame_type,
        tuple(sorted((role.name, role.value) for role in frame.roles)),
    )


def state_signature(state: State) -> tuple[str, str, str]:
    return state.name, state.left, state.right


def retime_frame(frame: Frame, time: int) -> Frame:
    frame_id = f"f{time}"
    return Frame(
        frame_id=frame_id,
        frame_type=frame.frame_type,
        time=time,
        roles=tuple(Role(frame_id, role.name, role.value) for role in frame.roles),
    )


def structure_with_frames_states(structure: Structure, frames: list[Frame], states: list[State]) -> Structure:
    return Structure(
        entities=structure.entities,
        rules=structure.rules,
        relations=tuple(state.to_relation() for state in states),
        events=tuple(frame.to_event() for frame in frames),
        query=structure.query,
        frames=tuple(frames),
        states=tuple(states),
        intentions=structure.intentions,
    )


def required_frame_role(frame: Frame, role_name: str) -> str:
    value = frame.role(role_name)
    if value is None:
        raise ParseError(f"Expected role {role_name} in frame {frame.frame_id}.")
    return value
